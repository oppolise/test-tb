# อธิบายโค้ด `loader.py` เชิงลึก

เอกสารนี้จะอธิบายการทำงานของไฟล์ `tb_plugin/torch_tb_profiler/profiler/loader.py` อย่างละเอียด เพื่อให้เข้าใจถึงวิธีการโหลด, ประมวลผล, และจัดการข้อมูลโปรไฟล์ โดยเฉพาะอย่างยิ่งในสถานการณ์ที่มีการทำงานแบบกระจาย (Distributed Training)

## ภาพรวมของ `loader.py`

`loader.py` มีคลาสหลักคือ `RunLoader` ซึ่งทำหน้าที่เป็น "ตัวโหลดข้อมูล" ของ profiler ทั้งหมด มันรับผิดชอบในการ:
1.  ค้นหาไฟล์ trace ที่ถูกสร้างโดย PyTorch Profiler ในไดเรกทอรีที่กำหนด
2.  จัดการไฟล์ trace ที่มาจากหลายๆ worker หรือหลายๆช่วงเวลา (span)
3.  ใช้ `multiprocessing` เพื่อประมวลผลไฟล์ trace แต่ละไฟล์แบบขนานกัน เพื่อความรวดเร็ว
4.  รวบรวมข้อมูลที่ประมวลผลแล้วจากแต่ละ process
5.  จัดการและประมวลผลข้อมูล communication สำหรับ distributed run โดยเฉพาะ
6.  คืนค่าเป็น `Run` object ที่มีข้อมูลโปรไฟล์ทั้งหมด พร้อมสำหรับนำไปแสดงผล

---

## การทำงานของคลาส `RunLoader`

### `__init__(self, name, run_dir, caches: io.Cache)`

*   **หน้าที่**: Constructor ของคลาส ใช้สำหรับตั้งค่าเริ่มต้น
*   **โค้ด**:
    ```python
    def __init__(self, name, run_dir, caches: io.Cache):
        self.run_name = name
        self.run_dir = run_dir
        self.caches = caches
        self.queue = Queue()
    ```
*   **คำอธิบาย**:
    *   `self.run_name`: (str) ชื่อของ "run" ที่กำลังจะโหลด เช่น "run/train-1"
    *   `self.run_dir`: (str) path ไปยังไดเรกทอรีที่เก็บไฟล์ trace ของ run นั้นๆ
    *   `self.caches`: (io.Cache) object ที่ใช้จัดการเรื่อง cache ของไฟล์ (อาจจะดึงไฟล์จาก remote storage และเก็บไว้ใน local cache)
    *   `self.queue`: (`multiprocessing.Queue`) สร้าง Queue สำหรับใช้สื่อสารระหว่าง process หลัก และ process ลูกที่ถูกสร้างขึ้นเพื่อประมวลผลไฟล์ trace Process ลูกจะส่งข้อมูลที่ประมวลผลเสร็จแล้วกลับมาให้ process หลักผ่าน Queue นี้

---

### `load(self)`

*   **หน้าที่**: เป็นเมธอดหลักที่เริ่มกระบวนการโหลดและประมวลผลทั้งหมด
*   **โค้ดและคำอธิบายเชิงลึก**:

    **ส่วนที่ 1: ค้นหาและจัดกลุ่มไฟล์ Trace**
    ```python
    workers = []
    spans_by_workers = defaultdict(list)
    for path in io.listdir(self.run_dir):
        if io.isdir(io.join(self.run_dir, path)):
            continue
        match = consts.WORKER_PATTERN.match(path)
        if not match:
            continue

        worker = match.group(1)
        span = match.group(2)
        if span is not None:
            # remove the starting dot (.)
            span = span[1:]
            bisect.insort(spans_by_workers[worker], span)

        workers.append((worker, span, path))
    ```
    *   **การทำงาน**:
        1.  `workers`: list ว่างสำหรับเก็บข้อมูลของแต่ละไฟล์ trace ที่เจอในรูปแบบ `(worker_name, span_name, file_path)`
        2.  `spans_by_workers`: dictionary ที่ใช้จัดกลุ่ม span (ช่วงเวลา) ของแต่ละ worker
        3.  `io.listdir(self.run_dir)`: วนลูปอ่านไฟล์ทั้งหมดในไดเรกทอรี `run_dir`
        4.  `consts.WORKER_PATTERN.match(path)`: ใช้ regular expression (`WORKER_PATTERN`) เพื่อตรวจสอบว่าชื่อไฟล์ตรงกับรูปแบบของไฟล์ trace หรือไม่ รูปแบบนี้มักจะดึงชื่อ worker และชื่อ span (ถ้ามี) ออกมาจากชื่อไฟล์ เช่น ไฟล์ชื่อ `worker1.1626338400.pt.trace.json` จะได้ `worker='worker1'` และ `span='1626338400'`
        5.  `bisect.insort(spans_by_workers[worker], span)`: หากไฟล์มี span, จะทำการเพิ่มชื่อ span เข้าไปใน list ของ worker นั้นๆ โดยใช้ `bisect.insort` เพื่อให้ list ของ span ถูกจัดเรียงตามลำดับเวลาอยู่เสมอ
        6.  `workers.append(...)`: เพิ่มข้อมูลของไฟล์ trace ที่เจอเข้า list `workers`

    **ส่วนที่ 2: สร้าง Index สำหรับ Span**
    ```python
    span_index_map = {}
    for worker, span_array in spans_by_workers.items():
        for i, span in enumerate(span_array, 1):
            span_index_map[(worker, span)] = i
    ```
    *   **การทำงาน**:
        1.  สำหรับแต่ละ worker ที่มีหลาย span, จะมีการสร้าง index (ลำดับที่) ให้กับแต่ละ span โดยเริ่มจาก 1
        2.  `span_index_map`: จะเก็บข้อมูลในรูปแบบ `(worker_name, span_name)` -> `index` เช่น `('worker1', '1626338400')` -> `1`
        3.  **ผลลัพธ์**: ทำให้สามารถอ้างอิงถึง span ด้วยตัวเลขลำดับได้ แทนที่จะใช้ timestamp ที่ยาว

    **ส่วนที่ 3: สร้างและเริ่ม Process ลูกเพื่อประมวลผลข้อมูล**
    ```python
    for worker, span, path in workers:
        # convert the span timestamp to the index.
        span_index = None if span is None else span_index_map[(worker, span)]
        p = Process(target=self._process_data, args=(worker, span_index, path))
        p.start()
    logger.info('started all processing')
    ```
    *   **การทำงาน**:
        1.  วนลูปผ่าน `workers` list (ข้อมูลของแต่ละไฟล์ trace)
        2.  `span_index = ...`: แปลงชื่อ span (timestamp) ให้เป็น index ที่สร้างไว้ในขั้นตอนก่อนหน้า
        3.  `p = Process(...)`: สร้าง process ใหม่ (`multiprocessing.Process`)
            *   `target=self._process_data`: กำหนดให้ process นี้ทำงานในฟังก์ชัน `_process_data`
            *   `args=(...)`: ส่ง argument ที่จำเป็น (ชื่อ worker, index ของ span, path ของไฟล์) ไปให้ฟังก์ชัน `_process_data`
        4.  `p.start()`: เริ่มการทำงานของ process ลูก process นี้จะทำงานแบบขนาน (asynchronously) กับ process หลัก

    **ส่วนที่ 4: รอรับข้อมูลจาก Process ลูกและรวบรวมผลลัพธ์**
    ```python
    distributed_run = Run(self.run_name, self.run_dir)
    run = Run(self.run_name, self.run_dir)
    num_items = len(workers)
    while num_items > 0:
        item: Tuple[RunProfile, DistributedRunProfileData] = self.queue.get()
        num_items -= 1
        r, d = item
        if r or d:
            logger.debug('Loaded profile via mp.Queue')
        if r is not None:
            run.add_profile(r)
        if d is not None:
            distributed_run.add_profile(d)
    ```
    *   **การทำงาน**:
        1.  สร้าง `Run` object สองอัน:
            *   `run`: สำหรับเก็บข้อมูลโปรไฟล์ทั่วไป
            *   `distributed_run`: สำหรับเก็บข้อมูลที่จำเป็นต่อการประมวลผลแบบ distributed โดยเฉพาะ
        2.  `num_items = len(workers)`: ตั้งค่าตัวนับให้เท่ากับจำนวน process ลูกที่สร้างขึ้น
        3.  `while num_items > 0`: วนลูปเพื่อรอรับข้อมูลจาก Queue จนกว่าจะครบทุก process
        4.  `item = self.queue.get()`: **ส่วนสำคัญ** - process หลักจะ "block" หรือหยุดรอที่บรรทัดนี้ จนกว่าจะมีข้อมูลถูกส่งเข้ามาใน Queue จาก process ลูก
        5.  เมื่อได้รับ `item` (ซึ่งเป็น tuple ของ `(RunProfile, DistributedRunProfileData)`), จะทำการแยกและเพิ่มข้อมูลเข้า `run` และ `distributed_run` ตามลำดับ

    **ส่วนที่ 5: ประมวลผลข้อมูล Distributed และคืนค่าสุดท้าย**
    ```python
    distributed_profiles = self._process_spans(distributed_run)
    for d in distributed_profiles:
        if d is not None:
            run.add_profile(d)

    # for no daemon process, no need to join them since it will automatically join
    return run
    ```
    *   **การทำงาน**:
        1.  `self._process_spans(distributed_run)`: เรียกใช้ฟังก์ชัน helper เพื่อประมวลผลข้อมูล distributed ที่รวบรวมมา (จะอธิบายในส่วนถัดไป)
        2.  ผลลัพธ์จาก `_process_spans` (ซึ่งเป็น `DistributedRunProfile`) จะถูกเพิ่มเข้าไปใน `run` object หลัก
        3.  `return run`: คืนค่า `Run` object ที่มีข้อมูลโปรไฟล์ทั้งหมดที่ผ่านการประมวลผลและรวบรวมเรียบร้อยแล้ว

---

### `_process_data(self, worker, span, path)`

*   **หน้าที่**: ฟังก์ชันนี้จะถูกรันใน process ลูกแต่ละตัว ทำหน้าที่ประมวลผลไฟล์ trace หนึ่งไฟล์
*   **โค้ด**:
    ```python
    def _process_data(self, worker, span, path):
        # ... (import absl.logging) ...
        try:
            logger.debug('Parse trace, run_dir=%s, worker=%s', self.run_dir, path)
            local_file = self.caches.get_remote_cache(io.join(self.run_dir, path))
            data = RunProfileData.parse(worker, span, local_file, self.caches.cache_dir)
            if data.trace_file_path != local_file:
                self.caches.add_file(local_file, data.trace_file_path)

            generator = RunGenerator(worker, span, data)
            profile = generator.generate_run_profile()
            dist_data = DistributedRunProfileData(data)

            logger.debug('Sending back profile via mp.Queue')
            self.queue.put((profile, dist_data))
        # ... (except KeyboardInterrupt, Exception) ...
    ```
*   **คำอธิบาย**:
    1.  `local_file = self.caches.get_remote_cache(...)`: ดึงไฟล์ trace มา (อาจจะผ่าน cache)
    2.  `data = RunProfileData.parse(...)`: **หัวใจของการประมวลผล** - เรียกใช้ `RunProfileData.parse()` (จากไฟล์ `data.py`) ซึ่งจะทำการ parse JSON, สร้าง event objects, สร้าง op tree, และเรียก parser เฉพาะทางทั้งหมด (gpu, kernel, memory, etc.) เพื่อประมวลผลข้อมูลในไฟล์นั้นๆ
    3.  `generator = RunGenerator(...)`: สร้าง `RunGenerator` (จาก `run_generator.py`)
    4.  `profile = generator.generate_run_profile()`: ใช้ `RunGenerator` เพื่อสร้าง `RunProfile` object ซึ่งเป็นข้อมูลที่จัดรูปแบบพร้อมสำหรับแสดงผลใน UI
    5.  `dist_data = DistributedRunProfileData(data)`: สร้าง `DistributedRunProfileData` object ซึ่งเก็บข้อมูลที่จำเป็นสำหรับการประมวลผลแบบ distributed (เช่น ข้อมูล communication)
    6.  `self.queue.put((profile, dist_data))`: **ส่งผลลัพธ์กลับ** - นำ `RunProfile` และ `DistributedRunProfileData` ที่สร้างเสร็จแล้ว ใส่เข้าไปใน Queue เพื่อให้ process หลักนำไปรวบรวมต่อไป

---

### `_process_spans(self, distributed_run: Run)`

*   **หน้าที่**: จัดการ run ที่มีหลาย span (ช่วงเวลา)
*   **โค้ด**:
    ```python
    def _process_spans(self, distributed_run: Run):
        spans = distributed_run.get_spans()
        if spans is None:
            return [self._process_distributed_profiles(distributed_run.get_profiles(), None)]
        else:
            span_profiles = []
            for span in spans:
                profiles = distributed_run.get_profiles(span=span)
                p = self._process_distributed_profiles(profiles, span)
                if p is not None:
                    span_profiles.append(p)
            return span_profiles
    ```
*   **คำอธิบาย**:
    *   ถ้า run ไม่มี span, จะเรียก `_process_distributed_profiles` เพียงครั้งเดียวสำหรับโปรไฟล์ทั้งหมด
    *   ถ้า run มีหลาย span, จะวนลูปผ่านแต่ละ span, ดึงโปรไฟล์ของ span นั้นๆ, และเรียก `_process_distributed_profiles` สำหรับแต่ละ span แยกกัน

---

### `_process_distributed_profiles(self, profiles: List[DistributedRunProfileData], span)`

*   **หน้าที่**: ประมวลผลข้อมูล communication ระหว่าง worker ต่างๆ ใน distributed run
*   **โค้ดและคำอธิบายเชิงลึก**:

    **ส่วนที่ 1: ตรวจสอบและรวบรวม Communication Node**
    ```python
    has_communication = True
    comm_node_lists: List[List[CommunicationNode]] = []
    for data in profiles:
        # ...
        if data.has_communication and data.comm_node_list:
            comm_node_lists.append(data.comm_node_list)
            if len(comm_node_lists[-1]) != len(comm_node_lists[0]):
                # ... (error logging) ...
                has_communication = False
        else:
            has_communication = False
    ```
    *   **การทำงาน**:
        1.  วนลูปผ่าน `profiles` (ซึ่งเป็น list ของ `DistributedRunProfileData` จากแต่ละ worker)
        2.  ตรวจสอบว่าแต่ละ worker มีข้อมูล communication (`data.has_communication`) หรือไม่ และจำนวน communication operation (`len(data.comm_node_list)`) เท่ากันทุก worker หรือไม่
        3.  หากไม่ตรงกัน จะตั้งค่า `has_communication = False` และจะข้ามการประมวลผลส่วนนี้ไป
        4.  หากตรงกัน จะรวบรวม `comm_node_list` (list ของ `CommunicationNode`) จากแต่ละ worker เก็บไว้ใน `comm_node_lists`

    **ส่วนที่ 2: คำนวณ "Real Communication Time"**
    ```python
    worker_num = len(comm_node_lists)
    for i, node in enumerate(comm_node_lists[0]):
        kernel_range_size = len(node.kernel_ranges)
        # loop for all communication kernel ranges in order
        for j in range(kernel_range_size):
            min_range = sys.maxsize
            # For each kernel_range, find the minist between workers as the real communication time
            for k in range(worker_num):
                # ... (error checking) ...
                if kernel_ranges:
                    if kernel_ranges[j][1] - kernel_ranges[j][0] < min_range:
                        min_range = kernel_ranges[j][1] - kernel_ranges[j][0]
            for k in range(worker_num):
                kernel_range = comm_node_lists[k][i].kernel_ranges[j]
                comm_node_lists[k][i].real_time_ranges.append((kernel_range[1] - min_range, kernel_range[1]))
    ```
    *   **การทำงาน (ซับซ้อนแต่สำคัญ)**:
        1.  วนลูปผ่าน communication operation แต่ละตัว (เช่น `all_reduce` ครั้งที่ `i`)
        2.  สำหรับ communication operation เดียวกัน, จะวนลูปผ่าน kernel แต่ละตัวที่ถูกเรียกโดย operation นั้น (เช่น `all_reduce` อาจเรียกหลาย kernel)
        3.  **หา `min_range`**: ในบรรดา worker ทั้งหมด, จะหาว่า worker ไหนใช้เวลาทำ kernel นี้ *สั้นที่สุด* (`min_range`)
        4.  **คำนวณ `real_time_ranges`**: แนวคิดคือ เวลาที่ใช้ในการสื่อสารข้อมูลจริงๆ (Data Transfer Time) คือเวลาที่สั้นที่สุดในบรรดา worker ทั้งหมด ส่วนเวลาที่เกินจากนั้นใน worker อื่นๆ จะถูกมองว่าเป็น "เวลาที่ใช้ในการรอ" (Wait/Sync Time)
        5.  `real_time_ranges` จะถูกคำนวณโดยเอาเวลาสิ้นสุดของ kernel (`kernel_range[1]`) ลบด้วย `min_range` เพื่อหาเวลาเริ่มต้นของ "real communication"
        6.  ข้อมูลนี้จะถูกใช้ใน `DistributedRunGenerator` เพื่อสร้างกราฟ Synchronizing/Communication Overview

    **ส่วนที่ 3: สร้าง Distributed Run Profile สุดท้าย**
    ```python
    for data in profiles:
        data.communication_parse()

    generator = DistributedRunGenerator(profiles, span)
    profile = generator.generate_run_profile()
    return profile
    ```
    *   **การทำงาน**:
        1.  `data.communication_parse()`: เรียกฟังก์ชันใน `communication.py` เพื่อคำนวณสถิติเพิ่มเติมจาก `real_time_ranges` ที่เพิ่งสร้าง
        2.  `generator = DistributedRunGenerator(...)`: สร้าง `DistributedRunGenerator`
        3.  `profile = generator.generate_run_profile()`: สร้าง `DistributedRunProfile` ซึ่งมีข้อมูลสำหรับ distributed view (เช่น กราฟ overlap, กราฟ wait time)
        4.  `return profile`: คืนค่าโปรไฟล์สำหรับ distributed run นี้

---
หวังว่าคำอธิบายนี้จะช่วยให้เข้าใจการทำงานของ `loader.py` ได้อย่างละเอียดและชัดเจนครับ
