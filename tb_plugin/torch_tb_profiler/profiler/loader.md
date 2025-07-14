# เอกสารอธิบายโค้ด `loader.py`

ไฟล์ `loader.py` เป็นส่วนประกอบสำคัญชิ้นแรกในกระบวนการประมวลผลข้อมูลของ PyTorch Profiler Plugin มันทำหน้าที่เป็น **"ผู้จัดการการโหลดข้อมูล"** โดยมีเป้าหมายหลักคือการค้นหาไฟล์ข้อมูลดิบ (`.pt.trace.json`) ที่ถูกสร้างขึ้นจากการโปรไฟล์, จัดการโหลดข้อมูลเหล่านั้นอย่างมีประสิทธิภาพโดยใช้หลายโปรเซส (multiprocessing) เพื่อไม่ให้หน้าเว็บของ TensorBoard ค้าง, และส่งต่อข้อมูลที่ประมวลผลเบื้องต้นไปยังส่วนอื่นๆ

---

## ภาพรวมการทำงาน

1.  **ค้นหาไฟล์:** สแกนหาไฟล์ `.pt.trace.json` ทั้งหมดในไดเรกทอรีของ run ที่กำหนด
2.  **แยกแยะ Worker และ Span:** แยกชื่อ "worker" (เช่น ชื่อเครื่อง, process ID) และ "span" (ช่วงเวลาที่โปรไฟล์) ออกจากชื่อไฟล์
3.  **สร้างโปรเซสย่อย:** สำหรับแต่ละไฟล์ที่พบ จะสร้างโปรเซสย่อย (child process) แยกกันเพื่อทำงานประมวลผลไฟล์นั้นๆ โดยเฉพาะ วิธีนี้ทำให้สามารถประมวลผลหลายไฟล์พร้อมกันได้ (parallel processing)
4.  **ประมวลผลในโปรเซสย่อย:**
    *   อ่านไฟล์ `.pt.trace.json`
    *   เรียกใช้ `RunProfileData.parse()` เพื่อแปลง JSON ให้เป็นอ็อบเจกต์ที่เก็บข้อมูล event ต่างๆ
    *   เรียกใช้ `RunGenerator` และ `DistributedRunGenerator` เพื่อสร้างอ็อบเจกต์ `RunProfile` และ `DistributedRunProfileData` ซึ่งเป็นข้อมูลที่พร้อมสำหรับการวิเคราะห์ในขั้นต่อไป
    *   ส่งผลลัพธ์กลับมายังโปรเซสหลักผ่าน `Queue`
5.  **รวบรวมผลลัพธ์:** โปรเซสหลักจะรอรับข้อมูลจาก `Queue` แล้วนำมารวบรวมเป็นอ็อบเจกต์ `Run` ที่สมบูรณ์
6.  **ประมวลผลข้อมูลแบบกระจาย (Distributed):** หากเป็นการโปรไฟล์แบบหลาย worker (distributed training) จะมีการประมวลผลเพิ่มเติมเพื่อวิเคราะห์การสื่อสาร (communication) ระหว่าง worker
7.  **ส่งคืนอ็อบเจกต์ `Run`:** คืนค่าอ็อบเจกต์ `Run` ที่บรรจุข้อมูลทั้งหมดที่ประมวลผลเสร็จสิ้นแล้ว เพื่อให้ `plugin.py` นำไปใช้งานต่อ

---

## การอธิบายโค้ดเชิงลึก

### `class RunLoader`

เป็นคลาสหลักของไฟล์นี้ ใช้จัดการการโหลดข้อมูลสำหรับหนึ่ง "run"

```python
class RunLoader:
    def __init__(self, name, run_dir, caches: io.Cache):
        self.run_name = name
        self.run_dir = run_dir
        self.caches = caches
        self.queue = Queue()
```

*   **`__init__(self, name, run_dir, caches)`**: Constructor ของคลาส
    *   `name`: ชื่อของ run (เช่น "run_2023_07_15")
    *   `run_dir`: พาธไปยังไดเรกทอรีของ run นั้นๆ
    *   `caches`: อ็อบเจกต์สำหรับจัดการแคชไฟล์ (อาจใช้ในกรณีที่ไฟล์ต้นฉบับอยู่บน remote storage)
    *   `self.queue = Queue()`: สร้าง `Queue` จาก `multiprocessing` ซึ่งเป็นช่องทางสื่อสารหลักสำหรับให้โปรเซสย่อยส่งข้อมูลกลับมายังโปรเซสหลัก

### `load(self)`

เมธอดหลักที่เริ่มต้นกระบวนการโหลดทั้งหมด

```python
    def load(self):
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

*   **บรรทัด 38-55**: ส่วนนี้จะวนลูปไฟล์ทั้งหมดใน `run_dir`
    *   `consts.WORKER_PATTERN.match(path)`: ใช้ Regular Expression เพื่อตรวจสอบว่าชื่อไฟล์ตรงกับรูปแบบของไฟล์ trace หรือไม่ (เช่น `worker_name.1626358800.pt.trace.json`)
    *   `worker = match.group(1)`: ดึงเอาชื่อ worker ออกมา
    *   `span = match.group(2)`: ดึงเอา timestamp (span) ออกมา
    *   `spans_by_workers`: เก็บรายการ span ของแต่ละ worker เพื่อนำไปเรียงลำดับและกำหนด index ในภายหลัง
    *   `workers.append(...)`: เก็บข้อมูล `(worker, span, path)` ของแต่ละไฟล์ที่พบ

```python
        span_index_map = {}
        for worker, span_array in spans_by_workers.items():
            for i, span in enumerate(span_array, 1):
                span_index_map[(worker, span)] = i
```

*   **บรรทัด 57-60**: สร้าง `span_index_map` เพื่อแปลงค่า timestamp (ที่เป็น string) ให้กลายเป็นลำดับตัวเลข (index) เช่น 1, 2, 3,... สำหรับแต่ละ worker ซึ่งจะช่วยให้การแสดงผลบน UI เรียงลำดับถูกต้อง

```python
        for worker, span, path in workers:
            # convert the span timestamp to the index.
            span_index = None if span is None else span_index_map[(worker, span)]
            p = Process(target=self._process_data, args=(worker, span_index, path))
            p.start()
        logger.info('started all processing')
```

*   **บรรทัด 62-66**: นี่คือหัวใจของการทำงานแบบขนาน (parallelism)
    *   วนลูป `workers` ที่หามาได้ในตอนแรก
    *   `p = Process(...)`: สร้างโปรเซสย่อยใหม่ โดยกำหนดให้ `target` คือเมธอด `_process_data` และส่ง `args` ที่จำเป็นเข้าไป (ชื่อ worker, span index, และพาธไฟล์)
    *   `p.start()`: เริ่มการทำงานของโปรเซสย่อย โปรแกรมจะไม่รอให้โปรเซสนี้ทำงานเสร็จ แต่จะวนลูปเพื่อสร้างและเริ่มโปรเซสถัดไปทันที

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

*   **บรรทัด 68-82**: ส่วนของการรวบรวมผลลัพธ์
    *   สร้างอ็อบเจกต์ `run` (สำหรับข้อมูลปกติ) และ `distributed_run` (สำหรับข้อมูล distributed) เปล่าๆ ขึ้นมารอ
    *   `while num_items > 0`: วนลูปเพื่อรอรับข้อมูลจากโปรเซสย่อยทั้งหมด
    *   `item = self.queue.get()`: **บรรทัดนี้จะหยุดรอ (block)** จนกว่าจะมีข้อมูลถูกส่งเข้ามาใน `Queue`
    *   เมื่อได้รับข้อมูล `item` (ซึ่งเป็น tuple ของ `RunProfile` และ `DistributedRunProfileData`) ก็จะนำไปเพิ่มในอ็อบเจกต์ `run` และ `distributed_run` ที่เตรียมไว้

```python
        distributed_profiles = self._process_spans(distributed_run)
        for d in distributed_profiles:
            if d is not None:
                run.add_profile(d)

        # for no daemon process, no need to join them since it will automatically join
        return run
```

*   **บรรทัด 84-89**:
    *   `self._process_spans(distributed_run)`: เรียกเมธอดเพื่อประมวลผลข้อมูล distributed เพิ่มเติม (จะอธิบายต่อไป)
    *   นำผลลัพธ์จากการประมวลผล distributed มาเพิ่มใน `run` หลัก
    *   `return run`: ส่งคืนอ็อบเจกต์ `Run` ที่มีข้อมูลทั้งหมดที่ประมวลผลเสร็จสมบูรณ์แล้ว

### `_process_data(self, worker, span, path)`

เมธอดนี้จะถูกรันใน **โปรเซสย่อย**

```python
    def _process_data(self, worker, span, path):
        # ... (logging setup) ...
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
        # ... (exception handling) ...
```

*   **บรรทัด 97**: `data = RunProfileData.parse(...)`: เรียกใช้ `parser` จากไฟล์ `data.py` เพื่ออ่านและแปลงเนื้อหาในไฟล์ JSON ให้กลายเป็นอ็อบเจกต์ `RunProfileData` ที่เก็บ event ต่างๆ ไว้อย่างเป็นระบบ
*   **บรรทัด 101-103**:
    *   `generator = RunGenerator(...)`: สร้าง `RunGenerator` เพื่อประมวลผลข้อมูลจาก `RunProfileData` ให้กลายเป็น `RunProfile` ซึ่งเป็นข้อมูลสรุปที่พร้อมแสดงผล (เช่น คำนวณค่าเฉลี่ย, สร้างข้อมูลสำหรับกราฟ)
    *   `dist_data = DistributedRunProfileData(data)`: สร้างอ็อบเจกต์สำหรับเก็บข้อมูลที่จำเป็นสำหรับการวิเคราะห์แบบ distributed
*   **บรรทัด 106**: `self.queue.put((profile, dist_data))`: **ส่งผลลัพธ์** ซึ่งเป็น tuple กลับไปยังโปรเซสหลักผ่าน `Queue`

### `_process_spans` และ `_process_distributed_profiles`

สองเมธอดนี้ทำงานร่วมกันเพื่อวิเคราะห์ข้อมูลในกรณีที่เป็น distributed training

*   **`_process_spans`**: จัดการข้อมูลที่มีหลาย `span` (หลายช่วงเวลา) โดยจะวนลูปเรียก `_process_distributed_profiles` สำหรับแต่ละ `span`
*   **`_process_distributed_profiles`**:
    *   ตรวจสอบว่า worker ทุกตัวมีข้อมูลการสื่อสาร (communication) หรือไม่ ถ้ามีไม่ครบ จะไม่สามารถแสดงผล distributed view ได้
    *   **หัวใจของการวิเคราะห์**: ส่วนที่ยากที่สุดคือการหา "เวลาที่แท้จริง" ของการสื่อสาร เช่น `all_reduce` ในการเทรนแบบ DDP นั้น kernel บน GPU ของทุก worker จะต้องทำงานพร้อมกัน เวลาที่ใช้จริงๆ จะถูกจำกัดโดย worker ที่ทำงานเสร็จช้าที่สุด โค้ดในส่วนนี้ (บรรทัด 137-150) จะทำการคำนวณหาช่วงเวลาที่ทับซ้อนกัน (overlap) และเวลาที่รอ (wait time) ของ communication operation
    *   สุดท้ายจะเรียก `DistributedRunGenerator` เพื่อสร้าง `DistributedRunProfile` ที่มีข้อมูลสรุปสำหรับ distributed view โดยเฉพาะ

---

## ผลลัพธ์ที่ได้

ผลลัพธ์สุดท้ายของไฟล์นี้คืออ็อบเจกต์ `Run` ที่ภายในบรรจุ `RunProfile` และ `DistributedRunProfile` ของทุก worker และทุก span ไว้อย่างครบถ้วน ข้อมูลเหล่านี้ถูกประมวลผลและจัดระเบียบมาอย่างดี พร้อมที่จะถูกดึงไปใช้โดย `plugin.py` เพื่อสร้างเป็น JSON response ส่งให้กับหน้าเว็บ TensorBoard แสดงผลเป็นกราฟและตารางที่ผู้ใช้เห็น
