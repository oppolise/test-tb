# อธิบายโค้ด `data.py` เชิงลึก

เอกสารนี้จะอธิบายการทำงานของไฟล์ `tb_plugin/torch_tb_profiler/profiler/data.py` อย่างละเอียด เพื่อให้เข้าใจถึงวิธีการที่ไฟล์นี้ทำหน้าที่เป็นศูนย์กลางในการประมวลผลข้อมูล (Data Processing Hub) ของ profiler

## ภาพรวมของ `data.py`

ไฟล์ `data.py` ประกอบด้วย 2 คลาสหลัก:
1.  **`RunProfileData`**: เป็นคลาสที่สำคัญที่สุด ทำหน้าที่เป็น "คอนเทนเนอร์" สำหรับข้อมูลโปรไฟล์ทั้งหมดที่มาจากไฟล์ trace *หนึ่งไฟล์* (ซึ่งแทน worker หนึ่งตัว ในช่วงเวลาหนึ่ง) และเป็น **ผู้ควบคุม (Orchestrator)** ที่เรียกใช้ parser ต่างๆ เพื่อประมวลผลข้อมูลดิบให้กลายเป็นข้อมูลเชิงลึกที่มีโครงสร้าง
2.  **`DistributedRunProfileData`**: เป็นคลาสที่เล็กกว่า ใช้สำหรับเก็บข้อมูลย่อยจาก `RunProfileData` ที่มีความจำเป็นต่อการประมวลผลในมุมมองแบบกระจาย (Distributed View) โดยเฉพาะข้อมูลเกี่ยวกับการสื่อสาร (Communication)

---

## การทำงานของคลาส `RunProfileData`

### `__init__(self, worker: str, span: str, trace_json: Dict)`

*   **หน้าที่**: Constructor ของคลาส ใช้สำหรับตั้งค่าเริ่มต้นและประมวลผลข้อมูลดิบจาก JSON trace เบื้องต้น
*   **โค้ดและคำอธิบายเชิงลึก**:

    **ส่วนที่ 1: ตั้งค่า Metadata และเตรียมตัวแปร**
    ```python
    def __init__(self, worker: str, span: str, trace_json: Dict):
        self.worker = worker
        self.span = span

        # metadatas
        self.is_pytorch_lightning = trace_json.get('Framework', None) == 'pytorch-lightning'
        self.data_schema_version = trace_json.get('schemaVersion', None)
        self.distributed_info = trace_json.get('distributedInfo', None)
        self.device_props = trace_json.get('deviceProperties', None)

        self.profiler_start_ts = float('inf')
        self.events: List[BaseEvent] = []
    ```
    *   `self.worker`, `self.span`: เก็บชื่อ worker และ span ที่ได้รับมา
    *   `metadatas`: ดึงข้อมูล metadata จาก `trace_json` เช่น `Framework`, `schemaVersion`, `distributedInfo`, `deviceProperties` ข้อมูลเหล่านี้มีประโยชน์ในการแสดงผลและวิเคราะห์ในภายหลัง
    *   `self.profiler_start_ts`: ตั้งค่า timestamp เริ่มต้นของ profiler เป็นค่าสูงสุด (infinity) เพื่อหาค่าที่น้อยที่สุดในภายหลัง
    *   `self.events`: สร้าง list ว่างสำหรับเก็บ event object ทั้งหมดที่จะถูกสร้างขึ้น

    **ส่วนที่ 2: วนลูปประมวลผล `traceEvents`**
    ```python
    trace_body = trace_json['traceEvents']
    fwd_bwd_events = []
    for data in trace_body:
        if data.get('cat') == 'fwdbwd':
            fwd_bwd_events.append(data)
        else:
            event = trace.create_event(data, self.is_pytorch_lightning)
            if event is not None:
                self.profiler_start_ts = min(self.profiler_start_ts, event.ts)
                self.events.append(event)
    ```
    *   **การทำงาน**:
        1.  ดึง list ของ event ทั้งหมดจาก `trace_json['traceEvents']`
        2.  สร้าง `fwd_bwd_events` list ว่างสำหรับเก็บ event ที่ใช้ในการเชื่อมโยง forward/backward pass โดยเฉพาะ
        3.  วนลูปผ่าน event แต่ละตัวใน `trace_body`:
            *   **ถ้า `cat` (category) เป็น `'fwdbwd'`**: event นี้จะถูกเก็บแยกไว้ใน `fwd_bwd_events` เพื่อนำไปใช้สร้าง `fwd_bwd_map` ในภายหลัง
            *   **ถ้าเป็น event อื่นๆ**:
                *   `event = trace.create_event(...)`: เรียกใช้ฟังก์ชันจาก `trace.py` เพื่อแปลงข้อมูลดิบ (`data`) ให้เป็น event object ที่เหมาะสม (เช่น `OperatorEvent`, `KernelEvent`)
                *   `self.profiler_start_ts = min(...)`: อัปเดต timestamp เริ่มต้นของ profiler ให้เป็นค่าที่น้อยที่สุดที่เคยเจอ
                *   `self.events.append(event)`: เพิ่ม event object ที่สร้างเสร็จแล้วเข้าไปใน list `self.events`

    **ส่วนที่ 3: จัดเรียง Event และสร้าง Forward-Backward Map**
    ```python
    self.events.sort(key=lambda e: e.ts)
    self.forward_backward_events = trace.create_association_events(fwd_bwd_events)
    ```
    *   `self.events.sort(...)`: จัดเรียง event ทั้งหมดใน `self.events` ตาม timestamp จากน้อยไปมาก ซึ่งสำคัญมากสำหรับการสร้าง Op Tree ที่ถูกต้อง
    *   `self.forward_backward_events = ...`: เรียกใช้ `create_association_events` จาก `trace.py` โดยส่ง `fwd_bwd_events` ที่เก็บไว้เข้าไป ผลลัพธ์ที่ได้คือ `fwd_bwd_map` (dictionary ที่เชื่อม timestamp ของ forward กับ backward) จะถูกเก็บไว้ในตัวแปรนี้

    **ส่วนที่ 4: ประกาศตัวแปรสำหรับเก็บผลลัพธ์จาก Parser**
    ```python
    # ... (ประกาศตัวแปรจำนวนมาก) ...
    self.tid2tree: Dict[int, OperatorNode] = None
    self.use_ddp: bool = False
    self.steps_costs = None
    self.gpu_metrics_parser: GPUMetricsParser = None
    self.op_list_groupby_name = None
    self.kernel_stat = None
    self.memory_snapshot: Optional[MemorySnapshot] = None
    self.recommendations = []
    ```
    *   **การทำงาน**: ส่วนนี้เป็นการประกาศตัวแปร instance ทั้งหมดที่จะถูกใช้เพื่อเก็บผลลัพธ์จากการประมวลผลของ parser ต่างๆ ในเมธอด `process()` ค่าเริ่มต้นจะเป็น `None` หรือค่าว่าง

### `@staticmethod parse(worker, span, path, cache_dir)`

*   **หน้าที่**: เป็น static method ที่ทำหน้าที่เป็น "entry point" สำหรับการสร้าง `RunProfileData` object จากไฟล์
*   **โค้ด**:
    ```python
    @staticmethod
    def parse(worker, span, path, cache_dir):
        trace_path, trace_json = RunProfileData._preprocess_file(path, cache_dir)

        profile = RunProfileData.from_json(worker, span, trace_json)
        profile.trace_file_path = trace_path
        return profile
    ```
*   **คำอธิบาย**:
    1.  `trace_path, trace_json = RunProfileData._preprocess_file(...)`: เรียกเมธอด helper `_preprocess_file` เพื่ออ่านและเตรียมข้อมูลจากไฟล์ trace (จะอธิบายต่อไป)
    2.  `profile = RunProfileData.from_json(...)`: เรียก `from_json` (จะอธิบายต่อไป) เพื่อสร้าง instance และทำการประมวลผลทั้งหมด
    3.  `profile.trace_file_path = trace_path`: เก็บ path ของไฟล์ trace ไว้
    4.  `return profile`: คืนค่า `RunProfileData` object ที่สร้างและประมวลผลเสร็จสมบูรณ์

### `@staticmethod from_json(worker, span, trace_json: Dict)`

*   **หน้าที่**: สร้าง instance และเรียกกระบวนการประมวลผลและวิเคราะห์
*   **โค้ด**:
    ```python
    @staticmethod
    def from_json(worker, span, trace_json: Dict):
        profile = RunProfileData(worker, span, trace_json)
        with utils.timing('Data processing'):
            profile.process()
        profile.analyze()
        return profile
    ```
*   **คำอธิบาย**:
    1.  `profile = RunProfileData(...)`: เรียก `__init__` เพื่อสร้าง instance และประมวลผลข้อมูลดิบเบื้องต้น
    2.  `profile.process()`: **เรียกเมธอด `process()` ซึ่งเป็นหัวใจหลักของการประมวลผลข้อมูล**
    3.  `profile.analyze()`: **เรียกเมธอด `analyze()` เพื่อสร้างคำแนะนำ (recommendations)**
    4.  `return profile`: คืนค่า instance ที่ประมวลผลและวิเคราะห์เสร็จแล้ว

### `@staticmethod _preprocess_file(trace_path, cache_dir)`

*   **หน้าที่**: อ่านไฟล์ trace, จัดการไฟล์บีบอัด (`.gz`), และแก้ไขปัญหาที่อาจเกิดขึ้นกับไฟล์ JSON
*   **โค้ด**:
    ```python
    # ... (โค้ดส่วนนี้ค่อนข้างยาวและจัดการหลายกรณี) ...
    ```
*   **คำอธิบาย**:
    *   **อ่านไฟล์**: อ่านข้อมูลจาก `trace_path`
    *   **Decompress**: หากไฟล์เป็น `.gz` จะทำการ decompress
    *   **จัดการ `JSONDecodeError`**:
        *   โค้ดมี work-around เพื่อจัดการกับไฟล์ JSON ที่อาจมี format ไม่ถูกต้อง (เช่น มี `N/A` ที่ไม่มีเครื่องหมายคำพูด) โดยพยายาม re-encode ใหม่
    *   **จัดการ "Record Window End"**: มี work-around เพื่อลบ event `Record Window End` ที่อาจมี timestamp ผิดปกติและทำให้ผลการวิเคราะห์เพี้ยน
    *   **เขียนไฟล์ชั่วคราว**: หากมีการแก้ไข JSON, จะเขียนผลลัพธ์ลงในไฟล์ `.gz` ชั่วคราวใน `cache_dir` และคืนค่า path ของไฟล์ใหม่นี้แทน
    *   **ผลลัพธ์**: คืนค่า `(trace_path, trace_json)` ที่พร้อมใช้งาน

### `process(self)`

*   **หน้าที่**: **เป็นเมธอดที่สำคัญที่สุด** ทำหน้าที่เป็นผู้ควบคุม (Orchestrator) เรียกใช้ parser ต่างๆ ตามลำดับเพื่อประมวลผล `self.events` และข้อมูลอื่นๆ ให้กลายเป็นข้อมูลเชิงลึก
*   **โค้ดและคำอธิบายเชิงลึก**:
    ```python
    def process(self):
        # 1. EventParser: สร้าง Op Tree และข้อมูลพื้นฐาน
        with utils.timing('EventParser.parse'):
            parser = EventParser()
            self.tid2tree, self.pl_tid2tree = parser.parse(self.events, self.forward_backward_events)

        # 2. คัดลอกผลลัพธ์จาก EventParser มาเก็บไว้
        self.has_runtime = parser.has_runtime
        # ... (และอื่นๆ เช่น has_kernel, use_ddp, steps_names, role_ranges) ...
        self.comm_node_list = parser.comm_node_list

        # 3. ModuleAggregator: รวมข้อมูล Operator
        with utils.timing('ModuleAggregator aggegation'):
            module_aggregator = ModuleAggregator()
            module_aggregator.aggregate(self.tid2tree)
        # ... (คัดลอกผลลัพธ์ op_list_groupby_name, kernel_list_groupby_name_op) ...

        # 4. OverallParser: คำนวณภาพรวม Step Costs
        with utils.timing('OverallParser aggegation'):
            overall_parser = OverallParser()
            overall_parser.aggregate(parser.steps, parser.role_ranges)
        self.avg_costs = overall_parser.avg_costs
        # ... (และอื่นๆ) ...

        # 5. GPUMetricsParser: วิเคราะห์การใช้ GPU
        self.gpu_metrics_parser = GPUMetricsParser.parse_events(...)

        # 6. TensorCoresParser: วิเคราะห์การใช้ Tensor Cores
        tensorcores_parser = TensorCoresParser.parse_events(...)
        # ... (คัดลอกผลลัพธ์) ...

        # 7. KernelParser: สรุปสถิติ Kernel (ถ้ามี)
        if self.has_kernel:
            kernel_parser = KernelParser()
            kernel_parser.parse_events(self.events)
            # ... (คัดลอกผลลัพธ์) ...

        # 8. MemoryParser: วิเคราะห์การใช้ Memory (ถ้ามี)
        memory_events = self._memory_events()
        if memory_events:
            memory_parser = MemoryParser(memory_events)
            self.memory_snapshot = memory_parser.find_memory_nodes(self.tid2tree)
    ```
*   **ลำดับการทำงาน**:
    1.  **`EventParser`**: ถูกเรียกเป็นอันดับแรกเพื่อสร้างโครงสร้างพื้นฐานที่สำคัญที่สุดคือ **Op Tree (`self.tid2tree`)** และข้อมูลพื้นฐานอื่นๆ เช่น `steps`, `role_ranges`
    2.  **`ModuleAggregator`**: ใช้ Op Tree ที่ได้มาเพื่อรวม (aggregate) ข้อมูล operator และ kernel สำหรับใช้ใน Operator View และ Kernel View
    3.  **`OverallParser`**: ใช้ `steps` และ `role_ranges` จาก `EventParser` เพื่อคำนวณภาพรวม performance
    4.  **`GPUMetricsParser`**, **`TensorCoresParser`**, **`KernelParser`**, **`MemoryParser`**: parser เฉพาะทางเหล่านี้ถูกเรียกใช้ตามลำดับเพื่อวิเคราะห์ข้อมูลในแต่ละแง่มุม และผลลัพธ์จะถูกเก็บไว้ในตัวแปร instance ของ `RunProfileData`

### `analyze(self)`

*   **หน้าที่**: ใช้ข้อมูลทั้งหมดที่ประมวลผลใน `process()` เพื่อสร้างคำแนะนำ (recommendations) ที่เป็นประโยชน์แก่ผู้ใช้
*   **โค้ด**:
    ```python
    def analyze(self):
        self.recommendations = []

        # ตัวอย่างการวิเคราะห์: DataLoader bottleneck
        dataloader_ratio = self.avg_costs.costs[ProfileRole.DataLoader] / self.avg_costs.costs[ProfileRole.Total]
        if dataloader_ratio > 0.05:
            # ... (สร้างข้อความแนะนำให้เพิ่ม num_workers) ...
            self.recommendations.append(...)

        self._analyze_distributed_metrics() # วิเคราะห์ DDP, DP, communication
        self._analyze_gpu_metrics()       # วิเคราะห์ GPU utilization

        if self.device_props:
            # วิเคราะห์ Tensor Core usage
            if (...):
                self.recommendations.append(...)

            # วิเคราะห์ Memory usage
            if self.memory_snapshot:
                if (...):
                    self.recommendations.append(...)
    ```
*   **คำอธิบาย**:
    *   เมธอดนี้เต็มไปด้วยตรรกะ `if-else` ที่ตรวจสอบเงื่อนไขต่างๆ จากข้อมูลที่ประมวลผลแล้ว
    *   ตัวอย่างเช่น:
        *   ถ้าเวลาที่ใช้ใน `DataLoader` สูงเกินไป -> แนะนำให้เพิ่ม `num_workers`
        *   ถ้าใช้ `DataParallel` (DP) แทนที่จะเป็น `DistributedDataParallel` (DDP) -> แนะนำให้เปลี่ยนไปใช้ DDP
        *   ถ้าสัดส่วนเวลา communication สูง -> แนะนำให้ลองใช้ Gradient Accumulation
        *   ถ้า GPU utilization ต่ำ -> แนะนำให้เพิ่ม batch size
        *   ถ้ามี op ที่รองรับ Tensor Core แต่ไม่ได้ใช้ -> แนะนำให้เปิดใช้งาน AMP (Automatic Mixed Precision)
        *   ถ้า memory ใกล้เต็ม -> แนะนำให้ใช้ Gradient Checkpointing
    *   คำแนะนำทั้งหมดจะถูกเก็บไว้ใน `self.recommendations` list

---

## การทำงานของคลาส `DistributedRunProfileData`

### `__init__(self, run_profile_data: RunProfileData)`

*   **หน้าที่**: คัดลอกข้อมูลที่จำเป็นจาก `RunProfileData` object สำหรับการประมวลผลแบบ distributed
*   **โค้ด**:
    ```python
    class DistributedRunProfileData:
        def __init__(self, run_profile_data: RunProfileData):
            self.worker = run_profile_data.worker
            self.span = run_profile_data.span
            self.steps_names = run_profile_data.steps_names
            self.has_communication = run_profile_data.has_communication
            self.comm_lib = run_profile_data.comm_lib
            self.comm_node_list = run_profile_data.comm_node_list
            # ... (และอื่นๆ) ...
    ```
*   **คำอธิบาย**:
    *   เป็นเพียงการคัดลอกข้อมูลจาก `RunProfileData` ที่ได้รับมา ไม่มีการประมวลผลใดๆ ใน `__init__`
    *   ข้อมูลที่สำคัญที่สุดที่คัดลอกมาคือ `self.comm_node_list` ซึ่งจะถูกใช้ใน `loader.py` เพื่อคำนวณ "Real Communication Time"

### `communication_parse(self)`

*   **หน้าที่**: เรียกใช้ `analyze_communication_nodes` เพื่อวิเคราะห์ข้อมูล communication เพิ่มเติม
*   **โค้ด**:
    ```python
    def communication_parse(self):
        self.step_comm_stats, self.total_comm_stats = analyze_communication_nodes(self.comm_node_list)
    ```
*   **คำอธิบาย**:
    *   เมธอดนี้จะถูกเรียกใน `loader.py` *หลังจาก* ที่ `real_time_ranges` ใน `comm_node_list` ถูกคำนวณและใส่เข้าไปแล้ว
    *   `analyze_communication_nodes` (จาก `communication.py`) จะคำนวณสถิติของ communication ในแต่ละ step และโดยรวม โดยใช้ `real_time_ranges` ที่ได้มา

---
หวังว่าคำอธิบายนี้จะช่วยให้เข้าใจบทบาทและการทำงานของ `data.py` ในฐานะศูนย์กลางการประมวลผลข้อมูลของ profiler ได้อย่างละเอียดและชัดเจนครับ
