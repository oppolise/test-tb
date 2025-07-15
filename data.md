# คำอธิบายโค้ด `profiler/data.py` เชิงลึก

เอกสารนี้อธิบายการทำงานของไฟล์ `data.py` จาก PyTorch Profiler Plugin อย่างละเอียด เพื่อให้เกิดความเข้าใจในวัตถุประสงค์, การไหลของข้อมูล, และตรรกะการทำงานของแต่ละส่วน สำหรับนำไปศึกษาและปรับใช้

## ภาพรวม

ไฟล์ `data.py` คือศูนย์กลางการประมวลผลข้อมูลของ profiler run หนึ่งๆ (สำหรับ worker หนึ่ง) ประกอบด้วยคลาสหลัก 2 คลาส:

1.  **`RunProfileData`**: ทำหน้าที่เป็นทั้ง Data Container และ Orchestrator สำหรับการประมวลผลข้อมูลของ worker หนึ่งๆ. มันรับข้อมูลดิบ, เรียกใช้ parser เฉพาะทางทั้งหมดตามลำดับ, และเก็บผลลัพธ์การวิเคราะห์ในทุกมิติ.
2.  **`DistributedRunProfileData`**: เป็น Data Container ที่เก็บข้อมูลที่จำเป็นจาก `RunProfileData` เพื่อใช้ในการสร้าง distributed view ในภายหลัง.

---

## `class RunProfileData`

คลาสนี้เก็บข้อมูลและผลการวิเคราะห์ทั้งหมดของ profiler run จาก worker หนึ่ง

### `__init__(self, worker: str, span: str, trace_json: Dict)`

Constructor ของคลาส รับข้อมูลดิบ (`trace_json`) และ metadata (`worker`, `span`) เข้ามาเพื่อเตรียมข้อมูลเบื้องต้นสำหรับการประมวลผล

*   **วัตถุประสงค์**:
    *   ตั้งค่า metadata ของ run.
    *   แปลง raw trace events (list of dictionaries) ให้เป็น list ของ typed event objects (`List[BaseEvent]`).
    *   สร้าง map ความสัมพันธ์ forward-backward.
*   **คำอธิบายโค้ด**:

    **ส่วนที่ 1: ตั้งค่า Metadata และประกาศ Instance Variables**
    ```python
    self.worker = worker
    self.span = span
    # เก็บ metadata อื่นๆ จาก trace_json เช่น framework, schema version, device properties
    self.is_pytorch_lightning = trace_json.get('Framework', None) == 'pytorch-lightning'
    self.data_schema_version = trace_json.get('schemaVersion', None)
    self.distributed_info = trace_json.get('distributedInfo', None)
    self.device_props = trace_json.get('deviceProperties', None)

    # ประกาศค่าเริ่มต้นสำหรับ instance variables ที่จะถูกเติมค่าในภายหลัง
    self.profiler_start_ts = float('inf')
    self.events: List[BaseEvent] = []
    self.tid2tree: Dict[int, OperatorNode] = None
    self.avg_costs = None
    # ... และอื่นๆ
    ```

    **ส่วนที่ 2: ประมวลผล Raw Trace Events**
    ```python
    trace_body = trace_json['traceEvents']
    fwd_bwd_events = []
    # วนลูป traceEvents ซึ่งเป็น list ของ event dictionary ดิบ
    for data in trace_body:
        # แยก event ที่ใช้สำหรับสร้างความสัมพันธ์ forward-backward โดยเฉพาะ
        if data.get('cat') == 'fwdbwd':
            fwd_bwd_events.append(data)
        else:
            # สำหรับ event อื่นๆ, เรียก trace.create_event() เพื่อแปลง dictionary ดิบ
            # ให้เป็น typed object (เช่น KernelEvent, OperatorEvent)
            event = trace.create_event(data, self.is_pytorch_lightning)
            if event is not None:
                # หา timestamp ที่เก่าที่สุดเพื่อเป็นจุดเริ่มต้นของ profiler
                self.profiler_start_ts = min(self.profiler_start_ts, event.ts)
                # เก็บ event object ที่สร้างได้
                self.events.append(event)
    ```

    **ส่วนที่ 3: จัดเรียง Events และสร้าง Fwd/Bwd Map**
    ```python
    # เรียง self.events ทั้งหมดตาม timestamp ซึ่งสำคัญมากสำหรับ parser ที่จะทำงานในลำดับถัดไป
    self.events.sort(key=lambda e: e.ts)
    # เรียก trace.create_association_events() เพื่อประมวลผล fwd_bwd_events
    # และสร้าง map ความสัมพันธ์ forward-backward (Dict[forward_ts, backward_ts])
    self.forward_backward_events = trace.create_association_events(fwd_bwd_events)
    ```
*   **ผลลัพธ์ของ `__init__`**: `RunProfileData` object จะมี `self.events` ที่เป็น list ของ event object ที่เรียงตามเวลา และ `self.forward_backward_events` ที่พร้อมใช้งาน.

### Static Methods: `parse`, `from_json`, `_preprocess_file`

กลุ่มของเมธอดที่ทำงานร่วมกันเพื่อเป็น entry point สำหรับการสร้าง `RunProfileData` object จากไฟล์.

*   **`parse(worker, span, path, cache_dir)`**:
    *   เป็นเมธอดที่ `RunLoader` เรียกใช้.
    *   เรียก `_preprocess_file` เพื่ออ่านและทำความสะอาดเนื้อหาไฟล์ JSON.
    *   เรียก `from_json` เพื่อสร้างและประมวลผล object.
*   **`_preprocess_file(trace_path, cache_dir)`**:
    *   จัดการกับการบีบอัดไฟล์ (`.gz`).
    *   **จัดการข้อผิดพลาดของ JSON**: มี try-except block เพื่อจัดการกับ JSON ที่อาจมี format ไม่ถูกต้อง (เช่น non-ASCII chars หรือ `N/A` ที่ไม่มี quote). หากเกิดข้อผิดพลาด จะพยายาม re-encode และแก้ไขข้อมูลแล้วโหลดอีกครั้ง.
    *   **จัดการ Timestamp ผิดปกติ**: ลบ event ชื่อ `'Record Window End'` ที่อาจมี timestamp ที่สูงผิดปกติและทำให้ visualization เพี้ยน.
    *   หากมีการแก้ไข JSON, จะเขียนข้อมูลที่แก้ไขแล้วลงในไฟล์ temp ใหม่และคืน path ของไฟล์ใหม่นั้น.
*   **`from_json(worker, span, trace_json: Dict)`**:
    *   สร้าง instance ของ `RunProfileData` จาก `trace_json` ที่ผ่านการ preprocess แล้ว.
    *   เรียก `profile.process()` และ `profile.analyze()` ซึ่งเป็นหัวใจหลักของการทำงาน.

### `process(self)`

เมธอดนี้สำคัญที่สุด ทำหน้าที่เป็น orchestrator เรียกใช้ parser เฉพาะทางทั้งหมดตามลำดับที่ถูกต้อง และเก็บผลลัพธ์ไว้ใน instance variables.

*   **วัตถุประสงค์**:
    *   แปลง list ของ events ให้เป็นข้อมูลเชิงลึกที่มีโครงสร้าง (structured insights).
    *   เติมค่า instance variables ทั้งหมดที่ประกาศไว้ใน `__init__`.
*   **ลำดับการทำงาน**:
    1.  **`EventParser.parse`**:
        ```python
        parser = EventParser()
        self.tid2tree, self.pl_tid2tree = parser.parse(self.events, self.forward_backward_events)
        ```
        *   เรียก `EventParser` ซึ่งเป็น parser ที่สำคัญที่สุด เพื่อสร้าง operator tree (`tid2tree`), แบ่ง step, ระบุ communication ops (`comm_node_list`), และจัดหมวดหมู่เวลาของ event (`role_ranges`).
        *   เก็บผลลัพธ์สำคัญๆ จาก `parser` ไว้ใน instance variables (`self.has_runtime`, `self.steps_names`, `self.used_devices`, `self.comm_node_list`, `self.role_ranges` ฯลฯ).

    2.  **`ModuleAggregator`**:
        ```python
        module_aggregator = ModuleAggregator()
        module_aggregator.aggregate(self.tid2tree)
        ```
        *   เรียก `ModuleAggregator` (จาก `op_agg.py`) เพื่อรวม (aggregate) ข้อมูล operator ตามชื่อและ input shape สำหรับใช้ใน Operator View. ผลลัพธ์ถูกเก็บใน `self.op_list_groupby_name`, `self.stack_lists_group_by_name` ฯลฯ.

    3.  **`OverallParser`**:
        ```python
        overall_parser = OverallParser()
        overall_parser.aggregate(parser.steps, parser.role_ranges)
        ```
        *   เรียก `OverallParser` เพื่อคำนวณสถิติภาพรวมของ step, ค่าเฉลี่ย, และที่สำคัญคือ **communication/computation overlap** โดยใช้ `role_ranges` จาก `EventParser`. ผลลัพธ์ถูกเก็บใน `self.avg_costs`, `self.steps_costs`, `self.comm_overlap_costs`.

    4.  **`GPUMetricsParser`**:
        *   เรียก `GPUMetricsParser.parse_events(...)` เพื่อคำนวณ GPU utilization และ SM efficiency จาก kernel events.

    5.  **`TensorCoresParser`**:
        *   เรียก `TensorCoresParser.parse_events(...)` เพื่อวิเคราะห์การใช้ Tensor Cores.

    6.  **`KernelParser`**:
        *   ถ้ามี kernel (`self.has_kernel`), เรียก `KernelParser.parse_events(...)` เพื่อสร้างสถิติของ kernel แต่ละตัว.

    7.  **`MemoryParser`**:
        *   ถ้ามี memory events, เรียก `MemoryParser(...)` และ `find_memory_nodes(...)` เพื่อวิเคราะห์การใช้หน่วยความจำและเชื่อมโยงกับ operator.

*   **ผลลัพธ์ของ `process()`**: `RunProfileData` object จะมีข้อมูลที่ผ่านการวิเคราะห์ในทุกมิติ พร้อมสำหรับนำไปใช้ต่อโดย `RunGenerator`.

### `analyze(self)`

เมธอดนี้สร้างคำแนะนำ (recommendations) ที่เป็นประโยชน์และแสดงใน UI.

*   **วัตถุประสงค์**:
    *   แปลงข้อมูลเชิงลึกที่ได้จากการ `process()` ให้เป็นคำแนะนำที่ผู้ใช้สามารถนำไปปฏิบัติได้.
*   **ตัวอย่างการวิเคราะห์**:
    *   `dataloader_ratio = self.avg_costs.costs[ProfileRole.DataLoader] / ...`: ตรวจสอบว่า dataloader เป็น bottleneck หรือไม่.
    *   `self.use_dp and len(self.used_devices) > 1`: แนะนำให้ใช้ DDP แทน DP.
    *   `self.tc_used_ratio == 0.0 and self.tc_eligible_ops_kernel_ratio > 0.0`: แนะนำให้เปิด AMP เพื่อใช้ Tensor Cores.
    *   `self.memory_snapshot.get_peak_memory() > total_mem * 0.9`: เตือนว่าการใช้ memory ใกล้เต็มความจุ.
    *   `communication_ratio > 0.1`: เตือนว่ามี overhead จาก communication สูง.
    *   `self.gpu_metrics_parser.gpu_utilization[gpu_id] < 0.5`: เตือนว่า GPU utilization ต่ำ.
*   **ผลลัพธ์**: `self.recommendations` ซึ่งเป็น list ของข้อความคำแนะนำ.

---

## `class DistributedRunProfileData`

เป็น Data Container ที่เก็บข้อมูลที่จำเป็นจาก `RunProfileData` เพื่อใช้ในการสร้าง distributed view.

*   **วัตถุประสงค์**:
    *   ลดความซับซ้อนโดยการดึงมาเฉพาะข้อมูลที่เกี่ยวข้องกับ distributed analysis.
    *   ทำหน้าที่เป็นที่พักข้อมูลก่อนที่จะถูกประมวลผลใน `RunLoader._process_distributed_profiles`.
*   **`__init__(self, run_profile_data: RunProfileData)`**:
    *   รับ `RunProfileData` object เข้ามา.
    *   คัดลอกเฉพาะ attributes ที่จำเป็นมาเก็บไว้ เช่น `worker`, `span`, `steps_names`, `has_communication`, `comm_node_list`, `comm_overlap_costs`.
    *   ประกาศ `self.total_comm_stats = None` และ `self.step_comm_stats = None` ซึ่งจะถูกเติมค่าในภายหลัง.
*   **`communication_parse(self)`**:
    *   **เมธอดนี้สำคัญมากสำหรับ distributed view**.
    *   มันจะถูกเรียกโดย `RunLoader` **หลังจากที่** `comm_node_list` ได้ถูกปรับแก้ (synchronized) แล้ว.
    *   เรียก `analyze_communication_nodes` (จาก `communication.py`) เพื่อคำนวณสถิติ communication โดยละเอียด (ต่อ step และต่อ op name) โดยใช้ `comm_node_list` ที่อัปเดตแล้ว (ซึ่งมี `real_time_ranges` ที่ถูกต้อง).
    *   ผลลัพธ์จะถูกเก็บใน `self.step_comm_stats` และ `self.total_comm_stats`.
