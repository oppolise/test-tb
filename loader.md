# คำอธิบายโค้ด `profiler/loader.py` เชิงลึก

เอกสารนี้อธิบายการทำงานของไฟล์ `loader.py` จาก PyTorch Profiler Plugin อย่างละเอียด เพื่อให้เกิดความเข้าใจในวัตถุประสงค์, การไหลของข้อมูล, และตรรกะการทำงานของแต่ละส่วน สำหรับนำไปศึกษาและปรับใช้

## ภาพรวม

ไฟล์ `loader.py` มีคลาสเดียวคือ `RunLoader` ซึ่งทำหน้าที่เป็นตัวจัดการหลัก (Orchestrator) ในการโหลดข้อมูลจากไดเรกทอรีของ profiler run หนึ่งๆ หน้าที่ของมันคือ:

1.  ค้นหาไฟล์ trace (`.pt.trace.json.gz`) ทั้งหมดที่อยู่ในไดเรกทอรี
2.  ประมวลผลไฟล์เหล่านั้นพร้อมกัน (concurrently) โดยใช้ `multiprocessing` เพื่อความรวดเร็ว
3.  รวบรวมผลลัพธ์ที่ได้จากการประมวลผล
4.  จัดการกับข้อมูลที่มาจากการทำ distributed training โดยเฉพาะ (synchronize communication data)
5.  ประกอบผลลัพธ์ทั้งหมดให้เป็น `Run` object ที่สมบูรณ์ ซึ่งจะถูกใช้โดย TensorBoard plugin เพื่อแสดงผลข้อมูลใน UI

---

## `class RunLoader`

### `__init__(self, name, run_dir, caches: io.Cache)`

Constructor ของคลาส ใช้เพื่อตั้งค่าเริ่มต้น

*   **วัตถุประสงค์**:
    *   เตรียมข้อมูลพื้นฐานที่ `RunLoader` ต้องใช้ในการทำงาน
*   **คำอธิบายโค้ด**:
    ```python
    # name: ชื่อของ run นี้ (เช่น "resnet50_ddp") ซึ่งจะแสดงใน UI
    self.run_name = name
    # run_dir: path ไปยังไดเรกทอรีที่มีไฟล์ trace ทั้งหมด
    self.run_dir = run_dir
    # caches: Cache object ที่รับเข้ามา ใช้สำหรับจัดการไฟล์ cache
    # (เช่น การดึงไฟล์จาก remote storage มาไว้ที่ local ก่อนประมวลผล)
    self.caches = caches
    # สร้าง multiprocessing.Queue ซึ่งเป็นช่องทางสื่อสารหลัก
    # สำหรับรับข้อมูลที่ประมวลผลเสร็จแล้วจาก child processes กลับมายัง main process
    self.queue = Queue()
    ```

---

### `load(self)`

เมธอดหลักของคลาสนี้ ทำหน้าที่ทั้งหมดตั้งแต่การค้นหาไฟล์ไปจนถึงการคืนค่า `Run` object ที่สมบูรณ์

#### ส่วนที่ 1: ค้นหาและจัดระเบียบไฟล์ Trace

*   **วัตถุประสงค์**:
    *   สำรวจไดเรกทอรี `run_dir` เพื่อหาไฟล์ trace ทั้งหมด
    *   แยกข้อมูล worker, span (ช่วงเวลาการ profile), และ path ของแต่ละไฟล์
    *   จัดระเบียบข้อมูล span เพื่อสร้าง index ที่เข้าใจง่าย
*   **คำอธิบายโค้ด**:
    ```python
    # workers: จะเก็บข้อมูลของไฟล์ trace ทั้งหมดในรูปแบบ list ของ tuple (worker_name, span_timestamp, file_path)
    workers = []
    # spans_by_workers: dictionary ที่เก็บ timestamp ของ span ทั้งหมดสำหรับแต่ละ worker
    spans_by_workers = defaultdict(list)

    # วนลูปไฟล์และไดเรกทอรีทั้งหมดใน self.run_dir
    for path in io.listdir(self.run_dir):
        # ข้ามถ้าเป็นไดเรกทอรี
        if io.isdir(io.join(self.run_dir, path)):
            continue
        # ใช้ Regular Expression (consts.WORKER_PATTERN) เพื่อจับคู่กับชื่อไฟล์ trace
        # เช่น "worker0.1623212756351.pt.trace.json.gz"
        match = consts.WORKER_PATTERN.match(path)
        if not match:
            continue

        # match.group(1) จะได้ชื่อ worker (เช่น "worker0")
        worker = match.group(1)
        # match.group(2) จะได้ span (เช่น ".1623212756351")
        span = match.group(2)
        if span is not None:
            # ลบจุด (.) นำหน้าออก
            span = span[1:]
            # เพิ่ม timestamp ของ span เข้าไปใน list ของ worker นั้นๆ และเรียงลำดับไปในตัว
            bisect.insort(spans_by_workers[worker], span)

        # เพิ่มข้อมูลของไฟล์นี้เข้าไปใน list `workers`
        workers.append((worker, span, path))

    # สร้าง Index สำหรับ Span
    # span_index_map: dictionary สำหรับแปลง timestamp ของ span ให้เป็น index ที่เรียงลำดับ (1, 2, 3, ...)
    span_index_map = {}
    for worker, span_array in spans_by_workers.items():
        # enumerate เริ่มจาก 1
        for i, span in enumerate(span_array, 1):
            span_index_map[(worker, span)] = i
    ```
*   **ผลลัพธ์ของส่วนนี้**:
    *   `workers`: list ที่มีข้อมูลของทุกไฟล์ trace
    *   `span_index_map`: dictionary ที่ใช้แปลง timestamp เป็น index ที่มนุษย์อ่านง่าย

#### ส่วนที่ 2: เริ่มต้น Multiprocessing

*   **วัตถุประสงค์**:
    *   สร้าง child process สำหรับแต่ละไฟล์ trace เพื่อประมวลผลพร้อมกัน
*   **คำอธิบายโค้ด**:
    ```python
    # วนลูปผ่าน list ของไฟล์ trace ทั้งหมด
    for worker, span, path in workers:
        # แปลง span timestamp เป็น index ที่สร้างไว้ (ถ้ามี)
        span_index = None if span is None else span_index_map[(worker, span)]
        # สร้าง Process ใหม่
        p = Process(target=self._process_data, args=(worker, span_index, path))
        # เริ่มการทำงานของ child process (ทำงานแบบ non-blocking)
        p.start()
    logger.info('started all processing')
    ```

#### ส่วนที่ 3: รวบรวมผลลัพธ์จาก Queue

*   **วัตถุประสงค์**:
    *   รอรับผลลัพธ์ที่ประมวลผลเสร็จแล้วจาก child processes
    *   รวบรวมข้อมูลลงใน `Run` object
*   **คำอธิบายโค้ด**:
    ```python
    # distributed_run: Run object สำหรับเก็บข้อมูลดิบของ distributed run เพื่อนำไปประมวลผลต่อ
    distributed_run = Run(self.run_name, self.run_dir)
    # run: Run object หลักสำหรับเก็บผลลัพธ์สุดท้ายที่จะส่งคืน
    run = Run(self.run_name, self.run_dir)
    num_items = len(workers)

    # วนลูปเพื่อรอรับข้อมูลจาก self.queue จนครบทุก process
    while num_items > 0:
        # self.queue.get() จะ block การทำงาน (รอ) จนกว่าจะมี item ถูกส่งเข้ามาใน queue
        item: Tuple[RunProfile, DistributedRunProfileData] = self.queue.get()
        num_items -= 1
        # แตก tuple ที่ได้รับออกมา
        r, d = item

        # เพิ่ม RunProfile ที่ประมวลผลเสร็จแล้ว (สำหรับ view ทั่วไป) เข้าไปใน Run object หลัก
        if r is not None:
            run.add_profile(r)
        # เพิ่ม DistributedRunProfileData (ข้อมูลดิบสำหรับ distributed view) เข้าไปใน Run object ที่ใช้สำหรับประมวลผล distributed
        if d is not None:
            distributed_run.add_profile(d)
    ```

#### ส่วนที่ 4: ประมวลผล Distributed View และส่งคืนค่า

*   **วัตถุประสงค์**:
    *   เรียกใช้การประมวลผลข้อมูลสำหรับ distributed view โดยเฉพาะ
    *   รวมผลลัพธ์สุดท้ายและส่งคืน `Run` object ที่สมบูรณ์
*   **คำอธิบายโค้ด**:
    ```python
    # เรียกเมธอดเพื่อประมวลผลข้อมูล distributed ที่รวบรวมมา
    distributed_profiles = self._process_spans(distributed_run)
    # วนลูปผลลัพธ์ (อาจมีหลาย distributed profile ถ้ามีหลาย span)
    for d in distributed_profiles:
        if d is not None:
            # เพิ่ม distributed profile ที่ประมวลผลเสร็จแล้วเข้าไปใน Run object หลัก
            run.add_profile(d)

    # ไม่จำเป็นต้อง join process เพราะเป็น non-daemon process ซึ่งจะถูก join อัตโนมัติเมื่อ main process จบการทำงาน
    return run
    ```

---

### `_process_data(self, worker, span, path)`

เมธอดนี้ทำงานใน child process เพื่อประมวลผลไฟล์ trace หนึ่งไฟล์

*   **วัตถุประสงค์**:
    *   ประมวลผลไฟล์ trace หนึ่งไฟล์อย่างสมบูรณ์
    *   ส่งผลลัพธ์กลับไปยัง main process ผ่าน Queue
*   **คำอธิบายโค้ด**:
    ```python
    try:
        # จัดการเรื่อง cache, ทำให้แน่ใจว่ามีไฟล์ trace อยู่ใน local filesystem
        local_file = self.caches.get_remote_cache(io.join(self.run_dir, path))
        # นี่คือการเรียกที่สำคัญมาก ไปยัง data.py เพื่อทำการ parse และประมวลผลไฟล์ trace ทั้งหมด
        # ซึ่งจะเรียก EventParser และ parser อื่นๆ ภายใน
        data = RunProfileData.parse(worker, span, local_file, self.caches.cache_dir)
        # จัดการ cache กรณีมีการ re-encode JSON
        if data.trace_file_path != local_file:
            self.caches.add_file(local_file, data.trace_file_path)

        # สร้าง RunGenerator object (จาก run_generator.py)
        generator = RunGenerator(worker, span, data)
        # เรียกเพื่อแปลง RunProfileData ให้เป็น RunProfile object ที่มีโครงสร้างเหมาะสำหรับ UI
        profile = generator.generate_run_profile()
        # สร้าง DistributedRunProfileData object ซึ่งเป็น subset ของข้อมูลที่จำเป็นสำหรับ distributed view
        dist_data = DistributedRunProfileData(data)

        # ส่งผลลัพธ์ (tuple) กลับไปยัง main process ผ่าน queue
        self.queue.put((profile, dist_data))
    except Exception as ex:
        # จัดการ error และส่ง None กลับไปเพื่อให้ main process ทำงานต่อได้
        logger.warning('Failed to parse profile data for Run %s on %s. Exception=%s',
                       self.run_name, worker, ex, exc_info=True)
        self.queue.put((None, None))
    ```

---

### `_process_spans(self, distributed_run: Run)`

เมธอดนี้เป็น wrapper สำหรับจัดการกับกรณีที่มีหลาย span

*   **วัตถุประสงค์**:
    *   แยกแยะระหว่างกรณีที่มี span เดียวกับหลาย span
    *   เรียก `_process_distributed_profiles` สำหรับแต่ละ span
*   **คำอธิบายโค้ด**:
    ```python
    spans = distributed_run.get_spans()
    # ถ้าไม่มี span (มีแค่ run เดียว)
    if spans is None:
        # เรียก _process_distributed_profiles ครั้งเดียวสำหรับ profile ทั้งหมด
        return [self._process_distributed_profiles(distributed_run.get_profiles(), None)]
    # ถ้ามีหลาย span
    else:
        span_profiles = []
        # วนลูปและเรียก _process_distributed_profiles สำหรับแต่ละ span
        for span in spans:
            profiles = distributed_run.get_profiles(span=span)
            p = self._process_distributed_profiles(profiles, span)
            if p is not None:
                span_profiles.append(p)
        return span_profiles
    ```

---

### `_process_distributed_profiles(self, profiles: List[DistributedRunProfileData], span)`

เมธอดนี้ประมวลผลข้อมูลจาก worker หลายๆ ตัวพร้อมกันเพื่อสร้าง distributed view profile

*   **วัตถุประสงค์**:
    *   ตรวจสอบความสอดคล้องของข้อมูล communication ระหว่าง workers
    *   ปรับแก้เวลา communication เพื่อแยก wait time ออกจาก data transfer time
    *   สร้าง `DistributedRunProfile` ที่สมบูรณ์
*   **คำอธิบายโค้ด**:

    **ส่วนที่ 1: ตรวจสอบความสอดคล้องของ Communication**
    ```python
    has_communication = True
    comm_node_lists: List[List[CommunicationNode]] = []
    # วนลูปผ่านข้อมูลจากแต่ละ worker
    for data in profiles:
        # ถ้า worker มีข้อมูล communication
        if data.has_communication and data.comm_node_list:
            comm_node_lists.append(data.comm_node_list)
            # ตรวจสอบว่าจำนวน communication op เท่ากับของ worker แรกหรือไม่
            if len(comm_node_lists[-1]) != len(comm_node_lists[0]):
                has_communication = False # จำนวนไม่เท่ากัน -> ปิด distributed view
        else:
            has_communication = False # บาง worker ไม่มี communication -> ปิด distributed view

    if not has_communication:
        return None # ไม่สร้าง distributed view
    ```

    **ส่วนที่ 2: ปรับแก้เวลา Communication (Synchronization)**
    ```python
    # นี่คือตรรกะที่ซับซ้อนและสำคัญ
    # แนวคิด: เวลา communication ที่แท้จริง (Data Transfer Time) ควรเท่ากับเวลาของ worker ที่ทำงานเสร็จเร็วที่สุด
    # ส่วนที่เกินจากนั้นใน worker อื่นๆ ถือเป็นเวลารอ (Wait Time)
    worker_num = len(comm_node_lists)
    # วนลูปตาม communication op (เช่น all_reduce ครั้งที่ 1, 2, 3...)
    for i, node in enumerate(comm_node_lists[0]):
        # วนลูปตาม kernel ของ communication op นั้น
        for j in range(len(node.kernel_ranges)):
            min_range = sys.maxsize
            # หา kernel ที่ทำงานสั้นที่สุดในบรรดา worker ทั้งหมดสำหรับ op เดียวกัน
            for k in range(worker_num):
                kernel_ranges = comm_node_lists[k][i].kernel_ranges
                # ตรวจสอบความสอดคล้องของจำนวน kernel
                if len(kernel_ranges) != len(node.kernel_ranges):
                    has_communication = False
                    return None
                # หา duration ที่สั้นที่สุด
                if kernel_ranges and (kernel_ranges[j][1] - kernel_ranges[j][0] < min_range):
                    min_range = kernel_ranges[j][1] - kernel_ranges[j][0]

            # คำนวณ real_time_ranges ใหม่สำหรับทุก worker โดยใช้ min_range ที่หาได้
            for k in range(worker_num):
                kernel_range = comm_node_lists[k][i].kernel_ranges[j]
                # real_time_ranges จะถูกใช้ใน DistributedRunGenerator เพื่อแยก Data Transfer Time และ Synchronizing Time
                comm_node_lists[k][i].real_time_ranges.append((kernel_range[1] - min_range, kernel_range[1]))
    ```

    **ส่วนที่ 3: คำนวณสถิติและสร้าง Profile สุดท้าย**
    ```python
    # เรียก data.communication_parse() บนทุก worker อีกครั้ง
    # เพื่อคำนวณสถิติ communication โดยละเอียดโดยใช้ real_time_ranges ที่เพิ่งปรับแก้ไป
    for data in profiles:
        data.communication_parse()

    # สร้าง DistributedRunGenerator object (จาก run_generator.py)
    generator = DistributedRunGenerator(profiles, span)
    # สร้าง DistributedRunProfile object ที่มีข้อมูลพร้อมสำหรับ UI
    profile = generator.generate_run_profile()
    return profile
    ```
