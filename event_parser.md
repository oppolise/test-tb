# คำอธิบายโค้ด `profiler/event_parser.py` เชิงลึก

เอกสารนี้อธิบายการทำงานของไฟล์ `event_parser.py` จาก PyTorch Profiler Plugin อย่างละเอียด เพื่อให้เกิดความเข้าใจในวัตถุประสงค์, การไหลของข้อมูล, และตรรกะการทำงานของแต่ละส่วน สำหรับนำไปศึกษาและปรับใช้

## ภาพรวม

ไฟล์ `event_parser.py` คือ **Engine หลักของการประมวลผล** ใน profiler plugin. หน้าที่ของมันคือการรับ flat list ของ `Event` objects (ที่สร้างโดย `trace.py`) มาแปลงให้เป็นข้อมูลที่มีโครงสร้างและมีความหมายเชิงลึก 2 อย่างหลักๆ คือ:

1.  **Operator Tree**: โครงสร้างแบบ parent-child ของ operations ทั้งหมดที่เกิดขึ้น, ซึ่งสะท้อน call stack การทำงาน.
2.  **Step-based Time Breakdown**: การแบ่งช่วงเวลาการทำงานออกเป็น "step" และจำแนกเวลาในแต่ละ step ว่าถูกใช้ไปกับกิจกรรมประเภทใด (`ProfileRole` เช่น Kernel, Communication, CPU, Memory).

ไฟล์นี้ใช้สถาปัตยกรรมแบบ Multiple Inheritance โดย `EventParser` สืบทอดความสามารถมาจาก `NodeParserMixin` และ `StepParser` เพื่อแยกความรับผิดชอบของโค้ดให้ชัดเจน.

---

## `class ProfileRole(IntEnum)`

*   **วัตถุประสงค์**: กำหนด Enum สำหรับประเภทของกิจกรรม (role) ที่เกิดขึ้นระหว่างการ profile. ใช้เป็น index สำหรับจัดเก็บและคำนวณเวลาของแต่ละประเภทอย่างเป็นระบบ.
*   **คำอธิบายโค้ด**:
    ```python
    class ProfileRole(IntEnum):
        Kernel = 0          # เวลาที่ใช้ใน GPU kernels (ที่ไม่ใช่ communication)
        Memcpy = 1          # เวลาที่ใช้ในการคัดลอกข้อมูล (H2D, D2H, D2D)
        Memset = 2          # เวลาที่ใช้ในการตั้งค่า memory
        Communication = 3   # เวลาที่ใช้ในการสื่อสาร (ทั้งบน CPU และ GPU)
        Runtime = 4         # เวลาที่ใช้ใน CUDA runtime calls (เช่น cudaLaunchKernel)
        DataLoader = 5      # เวลาที่ใช้ใน DataLoader
        CpuOp = 6           # เวลาที่ใช้ใน CPU operators (ที่ไม่ใช่ communication)
        Other = 7           # เวลาอื่นๆ ที่ไม่เข้าพวก
        Total = 8           # ใช้สำหรับระบุจำนวน role ทั้งหมด (ไม่ได้ใช้เก็บเวลา)
    ```

---

## `class NodeParserMixin`

Mixin นี้รับผิดชอบการแปลง `Event` objects ให้เป็น `Node` objects (จาก `node.py`) และสร้างความสัมพันธ์เบื้องต้นระหว่าง node เหล่านั้น.

### `__init__(self, *args, **kwargs)`

*   **วัตถุประสงค์**: ประกาศ instance variables ที่จะใช้เก็บผลลัพธ์จากการ parse node.
*   **คำอธิบายโค้ด**:
    ```python
    # Dict[external_id, CommunicationNode] - dictionary สำคัญที่ใช้เก็บ CommunicationNode ที่เจอ
    # เพื่อให้ kernel ที่มาทีหลังสามารถหา op ที่เป็นเจ้าของได้ผ่าน external_id
    self.communication_data: Dict[int, CommunicationNode] = {}
    # list ของ node ที่สร้างขึ้นทั้งหมด เพื่อใช้อ้างอิงในภายหลัง
    self.device_node_list: List[DeviceNode] = []
    self.runtime_node_list: List[RuntimeNode] = []
    # flags และ set สำหรับเก็บข้อมูลเกี่ยวกับสภาพแวดล้อมการทำงาน
    self.used_devices = set()
    self.use_dp = False
    self.use_ddp = False
    self.comm_lib = set()
    ```

### `parse_nodes(self, events: Iterable[BaseEvent])`

*   **วัตถุประสงค์**: เป็นเมธอดหลักของ mixin นี้. วนลูปผ่าน `events` ทั้งหมดเพื่อสร้างโครงสร้าง node เบื้องต้น.
*   **การทำงาน**:
    1.  **ประกาศ Temporary Data Structures**: สร้าง dictionaries หลายตัวเพื่อใช้เก็บข้อมูลชั่วคราวและสร้างความสัมพันธ์ระหว่างการ parse:
        *   `tid2list`: `Dict[int, List[OperatorNode]]` - เก็บ `OperatorNode` ทั้งหมด แยกตาม thread ID (TID). **นี่จะเป็น input หลักสำหรับสร้าง operator tree**.
        *   `corrid_to_device`: `Dict[correlation_id, List[DeviceNode]]` - ใช้ `correlation_id` เพื่อเชื่อม GPU kernel/memcpy/memset (`DeviceNode`) กับ CUDA runtime call ที่เรียกมัน.
        *   `corrid_to_runtime`: `Dict[correlation_id, RuntimeNode]` - เก็บ `RuntimeNode` โดยใช้ `correlation_id`.
        *   `externalid_to_runtime`: `Dict[external_id, List[RuntimeNode]]` - ใช้ `external_id` เพื่อเชื่อม `RuntimeNode` กับ CPU operator ที่เป็นคน launch.
    2.  **Loop 1 - `_parse_node`**: วนลูปผ่าน event ทั้งหมดและเรียก `_parse_node` เพื่อประมวลผลทีละ event.
    3.  **Loop 2 - `_update_communication_node` (ถ้าใช้ NCCL)**: วนลูป `KernelEvent` อีกครั้งเพื่อนำข้อมูลไปอัปเดต `CommunicationNode` ที่สร้างไว้ใน loop แรก.
    4.  **Associate Runtimes with CPU events**: วนลูป `tid2list` และใช้ `externalid_to_runtime` เพื่อนำ `RuntimeNode` ทั้งหมดไปใส่ใน `op.runtimes` ของ `OperatorNode` ที่เป็นเจ้าของ.
    5.  **Return**: คืนค่า `tid2list` และข้อมูลอื่นๆ ที่จำเป็นสำหรับ `EventParser`.

### `_parse_node(...)`

*   **วัตถุประสงค์**: ประมวลผล `Event` เดี่ยวๆ และสร้าง `Node` ที่เหมาะสม พร้อมทั้งเก็บเข้า temporary data structures.
*   **ตรรกะการทำงาน (สำคัญมาก)**:
    *   **`if event.type in [EventTypes.KERNEL, EventTypes.MEMCPY, EventTypes.MEMSET]`**:
        *   สร้าง `DeviceNode`.
        *   ถ้าเจอ `RuntimeNode` ที่มี `correlation_id` เดียวกันรออยู่แล้วใน `corrid_to_runtime`, ให้เพิ่ม `DeviceNode` นี้เข้าไปใน `rt_node.device_nodes`.
        *   ถ้าไม่, ให้เก็บ `DeviceNode` นี้ไว้ใน `corrid_to_device` เพื่อรอ `RuntimeNode` มาจับคู่ทีหลัง.
    *   **`elif event.type == EventTypes.RUNTIME`**:
        *   สร้าง `RuntimeNode`.
        *   พยายามดึง `DeviceNode` ที่มี `correlation_id` เดียวกันออกจาก `corrid_to_device` (ถ้ามี) และใส่เข้าไปใน `rt_node.device_nodes`.
        *   เก็บ `RuntimeNode` นี้ไว้ใน `corrid_to_runtime` และ `externalid_to_runtime`.
    *   **`elif event.type in [EventTypes.OPERATOR, ...]`**:
        *   สร้าง `OperatorNode` (หรือ subclass ของมันเช่น `ProfilerStepNode`, `ModuleNode`).
        *   **`if event.name in NcclOpNameSet or event.name in GlooOpNameSet:`**
            *   **นี่คือจุดที่ระบุ Communication Op.**
            *   สร้าง `CommunicationNode`.
            *   **เก็บ `CommunicationNode` ไว้ใน `self.communication_data` โดยใช้ `op_node.external_id` เป็น key.** นี่คือหัวใจของการเชื่อมโยง GPU kernel กับ communication op.
        *   เพิ่ม `op_node` ที่สร้างได้เข้าไปใน `tid2list` ตาม thread ID ของมัน.

### `_update_communication_node(self, event: KernelEvent)`

*   **วัตถุประสงค์**: อัปเดต `CommunicationNode` ด้วยข้อมูลจาก `KernelEvent` ที่เกี่ยวข้อง.
*   **การทำงาน**:
    *   `comm_node = self.communication_data.get(event.external_id)`: ใช้ `external_id` ของ kernel เพื่อค้นหา `CommunicationNode` ที่คู่กัน.
    *   ถ้าเจอ, `comm_node.kernel_ranges.append((ts, ts + dur))` และ `comm_node.total_time += dur`: เพิ่มช่วงเวลาและเวลารวมของ kernel เข้าไปใน `CommunicationNode`.

---

## `class StepParser`

Mixin นี้รับผิดชอบการวิเคราะห์ภาพรวมของเวลา, การแบ่งเป็น step, และการจัดหมวดหมู่เวลา.

### `__init__(self)`

*   **วัตถุประสงค์**: ประกาศ instance variables สำหรับเก็บผลลัพธ์.
*   **คำอธิบายโค้ด**:
    *   `self.role_ranges`: `List[List[Tuple[int, int]]]` - โครงสร้างข้อมูลหลัก, เป็น list ของ list. index แรกคือ `ProfileRole` (e.g., `self.role_ranges[ProfileRole.Kernel]`), และ list ภายในจะเก็บ tuple `(start_time, end_time)` ของทุก event ที่จัดอยู่ใน role นั้น.
    *   `self.steps`: `List[Tuple[int, int]]` - เก็บช่วงเวลาของแต่ละ profiler step.
    *   `self.steps_names`: `List[str]` - เก็บชื่อของแต่ละ step.

### `parse_steps(self, events: Iterable[DurationEvent], comm_nodes: Dict[int, CommunicationNode])`

*   **วัตถุประสงค์**: เป็นเมธอดหลักของ mixin นี้.
*   **การทำงาน**:
    1.  วนลูปผ่าน event ทั้งหมดเพื่อเรียก `_parse_step`.
    2.  หา `global_start_ts` และ `global_end_ts` จาก event `PyTorch Profiler (0)`.
    3.  ถ้าไม่เจอ `ProfilerStep#` event เลย, จะถือว่าทั้ง run เป็น step เดียว.
    4.  `for i in range(len(self.role_ranges)): self.role_ranges[i] = merge_ranges(self.role_ranges[i])`: สำหรับแต่ละ role, ทำการรวมช่วงเวลาที่ซ้อนทับกันให้เป็นช่วงเดียวที่ใหญ่ขึ้น.

### `_parse_step(self, event: DurationEvent, comm_nodes: Dict[int, CommunicationNode])`

*   **วัตถุประสงค์**: จัดประเภท `Event` เดี่ยวๆ เข้าไปใน `self.role_ranges`.
*   **ตรรกะการทำงาน (สำคัญมาก)**:
    *   `if evt_type == EventTypes.KERNEL:`
        *   `if event.external_id in comm_nodes:`: **นี่คือจุดที่จัดประเภท Kernel เข้าเป็น Communication.** ถ้า `external_id` ของ kernel อยู่ใน `comm_nodes` (ที่มาจาก `NodeParserMixin.communication_data`), ให้เพิ่มช่วงเวลาของ kernel นี้เข้า `self.role_ranges[ProfileRole.Communication]`.
        *   `else:`: ถ้าไม่, ให้เพิ่มเข้า `self.role_ranges[ProfileRole.Kernel]`.
    *   `elif evt_type == EventTypes.MEMCPY:`: เพิ่มเข้า `self.role_ranges[ProfileRole.Memcpy]`.
    *   ... (และอื่นๆ สำหรับ `MEMSET`, `RUNTIME`).
    *   `elif event.name.startswith('enumerate(DataLoader)#')`: เพิ่มเข้า `self.role_ranges[ProfileRole.DataLoader]`.
    *   `elif event.type == EventTypes.PROFILER_STEP:`: เพิ่มช่วงเวลาเข้า `self.steps` และชื่อเข้า `self.steps_names`.
    *   `elif evt_type in [EventTypes.PYTHON, EventTypes.OPERATOR, ...]`
        *   `if event.name in GlooOpNameSet or event.name in NcclOpNameSet:`: **นี่คือจุดที่จัดประเภท CPU Op เข้าเป็น Communication.** เพิ่มช่วงเวลาเข้า `self.role_ranges[ProfileRole.Communication]`.
        *   `else:`: ถ้าไม่, ให้เพิ่มเข้า `self.role_ranges[ProfileRole.CpuOp]`.

### `_find_device_steps` และ `_update_steps_duration`

*   **วัตถุประสงค์**: ปรับแก้ขอบเขตเวลาของ step (`self.steps`) ให้แม่นยำขึ้น. โดยปกติ step จะถูกกำหนดโดย CPU event (`ProfilerStep#...`), แต่อาจมี GPU kernel ที่ถูก launch ใน step นั้นทำงานเสร็จทีหลัง. เมธอดเหล่านี้จะหาเวลาสิ้นสุดของ kernel สุดท้ายที่ถูก launch ในแต่ละ step และขยายเวลาของ step นั้นให้ครอบคลุม.

---

## `class EventParser(NodeParserMixin, StepParser)`

คลาสหลักที่สืบทอดความสามารถจาก mixin ทั้งสอง และ orchestrate การทำงานทั้งหมด.

### `parse(self, events: Iterable[BaseEvent], fwd_bwd_map: Dict[int, int])`

*   **วัตถุประสงค์**: เป็น entry point หลักที่ `RunProfileData` เรียกใช้ เพื่อแปลง flat list ของ events ให้เป็นข้อมูลที่มีโครงสร้างสมบูรณ์.
*   **ลำดับการทำงาน**:
    1.  `tid2list, ... = self.parse_nodes(events)`: เรียก `NodeParserMixin` เพื่อสร้าง flat list ของ operator nodes (`tid2list`) และ `communication_data`.
    2.  `builder = OpTreeBuilder(); tid2tree = builder.build_tree(tid2list, ..., fwd_bwd_map=fwd_bwd_map)`: เรียก `OpTreeBuilder` (จาก `op_tree.py`) เพื่อแปลง `tid2list` ให้กลายเป็น operator tree ที่มีโครงสร้าง parent-child. ส่ง `fwd_bwd_map` เข้าไปด้วยเพื่อช่วยในการสร้าง tree.
    3.  `self.parse_steps(events, self.communication_data)`: เรียก `StepParser` เพื่อแบ่ง step และจัดหมวดหมู่เวลา (`role_ranges`).
    4.  `self.update_device_steps(self.runtime_node_list)`: ปรับแก้ขอบเขตของ step ตาม device events.
    5.  `self.comm_node_list = generate_communication_nodes(...)`: เรียกฟังก์ชันจาก `communication.py` เพื่อประมวลผล `communication_data` ที่รวบรวมมาให้เป็น `comm_node_list` ที่สมบูรณ์ (มีการจัดเรียงและกำหนด step name).
    6.  **Return `tid2tree`, `pl_tid2tree`**: คืนค่า operator tree ที่สมบูรณ์กลับไปให้ `RunProfileData`. (instance ของ `EventParser` เองก็จะเก็บผลลัพธ์อื่นๆ เช่น `self.steps`, `self.role_ranges`, `self.comm_node_list` ไว้ให้ `RunProfileData` ดึงไปใช้ต่อ).
