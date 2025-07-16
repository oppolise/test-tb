# อธิบายโค้ด `event_parser.py` เชิงลึก

เอกสารนี้จะอธิบายการทำงานของไฟล์ `tb_plugin/torch_tb_profiler/profiler/event_parser.py` อย่างละเอียด เพื่อให้เข้าใจถึงวิธีการที่ไฟล์นี้ทำหน้าที่เป็น **ตัวประมวลผล Event หลัก** ที่แปลงรายการ Event ดิบให้กลายเป็นข้อมูลที่มีโครงสร้างและมีความหมาย เช่น Op Tree, Step-wise performance breakdown

## ภาพรวมของ `event_parser.py`

ถ้า `data.py` คือ "ผู้ควบคุม" (Orchestrator) `event_parser.py` ก็เปรียบเสมือน **"หัวหน้าคนงาน" (Foreman)** ที่รับรายการ Event objects (จาก `trace.py`) มา แล้วจัดการแบ่งงาน, ประมวลผล, และสร้างผลลัพธ์พื้นฐานที่สำคัญ 2 อย่างคือ:
1.  **รายการของ Operator Nodes ที่จัดกลุ่มตาม Thread ID (`tid2list`)**: ซึ่งเป็นวัตถุดิบสำหรับสร้าง Op Tree
2.  **ข้อมูลสรุปของแต่ละ Step (`steps`, `role_ranges`)**: ซึ่งใช้สำหรับสร้างมุมมองภาพรวม (Overview)

ไฟล์นี้ประกอบด้วย 3 คลาสหลักที่ทำงานร่วมกัน:
1.  **`NodeParserMixin`**: ทำหน้าที่แปลง Event แต่ละตัวให้เป็น `Node` object ที่เหมาะสม (เช่น `OperatorNode`, `DeviceNode`, `RuntimeNode`, `CommunicationNode`) และเชื่อมโยงความสัมพันธ์เบื้องต้นระหว่าง Node เหล่านี้
2.  **`StepParser`**: ทำหน้าที่ระบุ "step" ของ profiler (เช่น `ProfilerStep#1`) และจัดหมวดหมู่เวลาที่ใช้ไปในแต่ละ step ตาม "บทบาท" (Role) เช่น Kernel, Communication, CPU Op
3.  **`EventParser`**: เป็นคลาสหลักที่สืบทอดคุณสมบัติจาก `NodeParserMixin` และ `StepParser` และเป็นผู้ควบคุมการทำงานทั้งหมดในไฟล์นี้

---

## การทำงานของโค้ดแต่ละส่วน

### นิยามพื้นฐาน

```python
CommLibTypes = IntEnum('CommLibTypes', ['Nccl', 'Gloo'], start=0)

class ProfileRole(IntEnum):
    Kernel = 0
    Memcpy = 1
    Memset = 2
    Communication = 3
    Runtime = 4
    DataLoader = 5
    CpuOp = 6
    Other = 7
    Total = 8
```
*   **`CommLibTypes`**: สร้าง enum สำหรับระบุประเภทของ communication library (NCCL หรือ Gloo)
*   **`ProfileRole`**: สร้าง enum ที่สำคัญมากสำหรับระบุ "บทบาท" ของการทำงานในช่วงเวลาต่างๆ ซึ่งจะถูกใช้ใน `StepParser` เพื่อแบ่งเวลาในแต่ละ step ออกเป็นส่วนๆ เช่น เวลาที่ใช้ใน Kernel, เวลาที่ใช้ใน Communication เป็นต้น

### `class NodeParserMixin`

คลาสนี้รับผิดชอบการแปลง event เป็น node และสร้างความสัมพันธ์ระหว่าง node

#### `__init__(self, *args, **kwargs)`
*   **หน้าที่**: Constructor, ประกาศตัวแปรสำหรับเก็บข้อมูลที่ได้จากการ parse node
*   **โค้ด**:
    ```python
    def __init__(self, *args, **kwargs):
        # ...
        self.communication_data: Dict[int, CommunicationNode] = {}
        self.device_node_list: List[DeviceNode] = []
        self.runtime_node_list: List[RuntimeNode] = []
        self.used_devices = set()
        self.use_dp = False
        self.use_ddp = False
        self.comm_lib = set()
    ```
*   **คำอธิบาย**:
    *   `self.communication_data`: Dictionary สำหรับเก็บ `CommunicationNode` โดยมี key เป็น `external_id` ของ operator ที่สร้างมันขึ้นมา
    *   `self.device_node_list`, `self.runtime_node_list`: List สำหรับเก็บ `DeviceNode` และ `RuntimeNode` ทั้งหมดที่เจอ
    *   `self.used_devices`: Set ของ ID ของ GPU ที่ถูกใช้งาน
    *   `self.use_dp`, `self.use_ddp`: Flag ที่บอกว่ามีการใช้ `DataParallel` หรือ `DistributedDataParallel` หรือไม่
    *   `self.comm_lib`: Set ที่เก็บประเภทของ communication library ที่ใช้ (NCCL, Gloo)

#### `parse_nodes(self, events: Iterable[BaseEvent])`
*   **หน้าที่**: เป็นเมธอดหลักของ Mixin นี้ วนลูปผ่าน event ทั้งหมดเพื่อเรียก `_parse_node` และจัดการความสัมพันธ์ของ Node หลังจากวนลูปเสร็จ
*   **โค้ดและคำอธิบายเชิงลึก**:

    **ส่วนที่ 1: เตรียมตัวแปรสำหรับเก็บข้อมูลชั่วคราว**
    ```python
    def parse_nodes(self, events: Iterable[BaseEvent]):
        # ...
        tid2list: Dict[int, List[OperatorNode]] = defaultdict(list)
        # ... (pl_tid2list, tid2zero_rt_list) ...
        corrid_to_device: Dict[int, List[DeviceNode]] = defaultdict(list)
        corrid_to_runtime: Dict[int, RuntimeNode] = {}
        externalid_to_runtime: Dict[int, List[RuntimeNode]] = defaultdict(list)
    ```
    *   **การทำงาน**: สร้าง dictionary ต่างๆ เพื่อใช้เป็น "ที่พัก" ข้อมูลระหว่างการประมวลผล:
        *   `tid2list`: **สำคัญมาก** - เป็นผลลัพธ์หลักอันหนึ่ง ใช้เก็บรายการ `OperatorNode` โดยจัดกลุ่มตาม Thread ID (`tid`)
        *   `corrid_to_device`: ใช้ `correlation_id` เพื่อเชื่อม `DeviceNode` (เช่น Kernel) กับ `RuntimeNode` (เช่น `cudaLaunchKernel`) ที่เรียกมัน
        *   `externalid_to_runtime`: ใช้ `external_id` เพื่อเชื่อม `RuntimeNode` กับ `OperatorNode` (CPU op) ที่เป็นต้นเหตุของการเรียก runtime นั้น

    **ส่วนที่ 2: วนลูปประมวลผล Event**
    ```python
    for event in events:
        if event.type == EventTypes.MEMORY:
            continue
        self._parse_node(event, corrid_to_device, ...)
    ```
    *   **การทำงาน**: วนลูปผ่าน `events` ทั้งหมดที่ได้รับมา (ยกเว้น Memory event) และส่งแต่ละ event ไปให้ `_parse_node` ประมวลผล

    **ส่วนที่ 3: อัปเดต Communication Node (ถ้าจำเป็น)**
    ```python
    if CommLibTypes.Nccl in self.comm_lib:
        for event in events:
            if event.type == EventTypes.KERNEL:
                self._update_communication_node(event)
    ```
    *   **การทำงาน**: หากมีการใช้ NCCL, จะวนลูปผ่าน event อีกครั้งเพื่อหา Kernel ที่เกี่ยวข้องกับ communication operation และอัปเดตข้อมูลใน `CommunicationNode` (จะอธิบายใน `_update_communication_node`)

    **ส่วนที่ 4: เชื่อมโยง Runtime Node กับ Operator Node**
    ```python
    for op_list in tid2list.values():
        for op in op_list:
            runtime_nodes = externalid_to_runtime.pop(op.external_id, [])
            if runtime_nodes:
                op.runtimes.extend(runtime_nodes)
    ```
    *   **การทำงาน**: วนลูปผ่าน `OperatorNode` ทั้งหมดที่สร้างขึ้น และใช้ `op.external_id` ไปค้นหา `RuntimeNode` ที่สอดคล้องกันใน `externalid_to_runtime` แล้วเพิ่ม `RuntimeNode` เหล่านั้นเข้าไปใน `op.runtimes` list
    *   **ผลลัพธ์**: ทำให้ `OperatorNode` แต่ละตัว "รู้" ว่ามันได้เรียก runtime อะไรไปบ้าง

    **ส่วนที่ 5: คืนค่าผลลัพธ์**
    ```python
    # ... (จัดการ staled_device_nodes) ...
    return tid2list, tid2zero_rt_list, staled_device_nodes, pl_tid2list
    ```
    *   **การทำงาน**: คืนค่า `tid2list` (รายการ OperatorNode จัดกลุ่มตาม Thread ID) และข้อมูลอื่นๆ ที่ `EventParser.parse()` จะนำไปใช้สร้าง Op Tree ต่อไป

#### `_parse_node(self, event: DurationEvent, ...)`
*   **หน้าที่**: **หัวใจของ `NodeParserMixin`** - รับ event หนึ่งตัวและตัดสินใจว่าจะสร้าง Node ประเภทใด และจะเก็บไว้ที่ไหน
*   **โค้ดและคำอธิบายเชิงลึก** (แบ่งตามประเภท event):
    *   **ถ้าเป็น Device Event (`KERNEL`, `MEMCPY`, `MEMSET`)**:
        ```python
        if event.type in [EventTypes.KERNEL, EventTypes.MEMCPY, EventTypes.MEMSET]:
            device_node = DeviceNode.create(event)
            if corrid in corrid_to_runtime:
                # ... (เพิ่ม device_node เข้าไปใน rt_node.device_nodes) ...
            else:
                corrid_to_device[corrid].append(device_node)
            self.device_node_list.append(device_node)
        ```
        1.  สร้าง `DeviceNode` จาก event
        2.  ใช้ `correlation_id` (`corrid`) เพื่อพยายามเชื่อมกับ `RuntimeNode` ที่อาจจะถูกสร้างไว้ก่อนแล้ว
        3.  ถ้าเจอ `RuntimeNode` ที่คู่กัน, ก็จะเพิ่ม `DeviceNode` นี้เข้าไปใน `rt_node.device_nodes`
        4.  ถ้ายังไม่เจอ, จะเก็บ `DeviceNode` นี้ไว้ใน `corrid_to_device` เพื่อรอให้ `RuntimeNode` ที่คู่กันมาเจอในภายหลัง
    *   **ถ้าเป็น Runtime Event (`RUNTIME`)**:
        ```python
        elif event.type == EventTypes.RUNTIME:
            device_nodes = corrid_to_device.pop(corrid, None)
            rt_node = RuntimeNode.create(event, device_nodes)
            corrid_to_runtime[corrid] = rt_node
            externalid_to_runtime[rt_node.external_id].append(rt_node)
            # ...
        ```
        1.  ใช้ `correlation_id` (`corrid`) ไปค้นหา `DeviceNode` ที่อาจจะถูกสร้างไว้ก่อนแล้วใน `corrid_to_device`
        2.  สร้าง `RuntimeNode` พร้อมกับ `device_nodes` ที่หาเจอ
        3.  เก็บ `RuntimeNode` ที่สร้างเสร็จแล้วไว้ใน `corrid_to_runtime` (เพื่อให้ `DeviceNode` ที่มาทีหลังหาเจอ) และ `externalid_to_runtime` (เพื่อให้ `OperatorNode` หาเจอ)
    *   **ถ้าเป็น Operator Event (`OPERATOR`, `PYTHON`, `MODULE`, etc.)**:
        ```python
        elif event.type in [...]:
            op_node = create_operator_node(event) # หรือ ProfilerStepNode.create, ModuleNode.create
            if event.name in NcclOpNameSet or event.name in GlooOpNameSet:
                comm_node = CommunicationNode.create(event)
                # ... (ตั้งค่า comm_node) ...
                self.communication_data[op_node.external_id] = comm_node
            if event.name == 'DataParallel.forward':
                self.use_dp = True
            if op_node:
                tid2list[int(tid)].append(op_node)
        ```
        1.  สร้าง `OperatorNode` (หรือคลาสลูก) ที่เหมาะสม
        2.  **ตรวจสอบว่าเป็น Communication Op หรือไม่**: ถ้าชื่อ event อยู่ใน `NcclOpNameSet` หรือ `GlooOpNameSet`, จะสร้าง `CommunicationNode` และเก็บไว้ใน `self.communication_data`
        3.  **ตรวจสอบ DP/DDP**: ถ้าเจอ op ชื่อ `DataParallel.forward` หรือ `DistributedDataParallel.forward`, จะตั้งค่า flag `self.use_dp` หรือ `self.use_ddp`
        4.  เพิ่ม `op_node` ที่สร้างเสร็จแล้วเข้าไปใน `tid2list` ของ thread ID นั้นๆ

#### `_update_communication_node(self, event: KernelEvent)`
*   **หน้าที่**: (สำหรับ NCCL) เชื่อมโยง Kernel event กับ `CommunicationNode` ที่ถูกสร้างไปแล้ว
*   **การทำงาน**: ใช้ `event.external_id` (ของ Kernel) ไปค้นหา `CommunicationNode` ที่สอดคล้องกันใน `self.communication_data` และเพิ่มช่วงเวลาของ kernel นี้เข้าไปใน `comm_node.kernel_ranges`

### `class StepParser`

คลาสนี้รับผิดชอบการวิเคราะห์ภาพรวมในแต่ละ step

#### `__init__(self)`
*   **หน้าที่**: Constructor, ประกาศตัวแปรสำหรับเก็บข้อมูลเกี่ยวกับ step และ role
*   **โค้ด**:
    ```python
    def __init__(self):
        self.role_ranges: List[List[Tuple[int, int]]] = [[] for _ in range(ProfileRole.Total - 1)]
        self.steps: List[Tuple[int, int]] = []
        self.steps_names: List[str] = []
        # ... (ตัวแปรเก็บ min/max timestamp) ...
    ```
*   **คำอธิบาย**:
    *   `self.role_ranges`: **สำคัญมาก** - เป็น list ของ list (`List[List]`) ที่ใช้เก็บ "ช่วงเวลา" (time ranges) ของการทำงานแต่ละประเภท (`ProfileRole`) เช่น `self.role_ranges[ProfileRole.Kernel]` จะเป็น list ของ `(start_time, end_time)` ทั้งหมดของ Kernel events
    *   `self.steps`: List ของ `(start_time, end_time)` ของแต่ละ profiler step
    *   `self.steps_names`: List ของชื่อ step (เช่น '1', '2')

#### `parse_steps(self, events: Iterable[DurationEvent], comm_nodes: Dict[int, CommunicationNode])`
*   **หน้าที่**: วนลูปผ่าน event ทั้งหมดเพื่อเรียก `_parse_step` และทำการสรุปผลหลังจากวนลูปเสร็จ
*   **การทำงาน**:
    1.  วนลูปผ่าน `events` และเรียก `self._parse_step` สำหรับแต่ละ event
    2.  หลังจากวนลูปเสร็จ, จะทำการ `merge_ranges` สำหรับแต่ละ role ใน `self.role_ranges` เพื่อรวมช่วงเวลาที่ติดกันให้เป็นช่วงเดียว

#### `_parse_step(self, event: DurationEvent, comm_nodes: Dict[int, CommunicationNode])`
*   **หน้าที่**: **หัวใจของ `StepParser`** - รับ event หนึ่งตัวและจัดหมวดหมู่ช่วงเวลาของมันลงใน `self.role_ranges` และ `self.steps`
*   **โค้ดและคำอธิบายเชิงลึก**:
    ```python
    def _parse_step(self, event: DurationEvent, comm_nodes: Dict[int, CommunicationNode]):
        ts = event.ts
        dur = event.duration
        evt_type = event.type
        if evt_type == EventTypes.KERNEL:
            if event.external_id in comm_nodes:
                self.role_ranges[ProfileRole.Communication].append((ts, ts + dur))
            else:
                self.role_ranges[ProfileRole.Kernel].append((ts, ts + dur))
        elif evt_type == EventTypes.MEMCPY:
            self.role_ranges[ProfileRole.Memcpy].append((ts, ts + dur))
        # ... (elif สำหรับ MEMSET, RUNTIME, DATALOADER) ...
        elif event.type == EventTypes.PROFILER_STEP:
            self.steps.append((ts, ts + dur))
            self.steps_names.append(str(event.step))
        elif evt_type in [EventTypes.PYTHON, EventTypes.OPERATOR, ...]:
            if event.name in GlooOpNameSet or event.name in NcclOpNameSet:
                self.role_ranges[ProfileRole.Communication].append((ts, ts + dur))
            else:
                self.role_ranges[ProfileRole.CpuOp].append((ts, ts + dur))
        # ...
    ```
    *   **การทำงาน**: ใช้ `if-elif` ขนาดใหญ่เพื่อตรวจสอบประเภทของ event:
        *   ถ้าเป็น `KERNEL`: จะตรวจสอบต่อว่า kernel นี้เกี่ยวข้องกับ communication หรือไม่ (โดยดูจาก `event.external_id` ว่าอยู่ใน `comm_nodes` หรือไม่) ถ้าใช่ จะเพิ่มช่วงเวลา vào `ProfileRole.Communication` ถ้าไม่ใช่ จะเพิ่ม vào `ProfileRole.Kernel`
        *   ถ้าเป็น `MEMCPY`, `MEMSET`, `RUNTIME`, `DATALOADER`: จะเพิ่มช่วงเวลา vào role ที่ตรงตัว
        *   ถ้าเป็น `PROFILER_STEP`: จะเพิ่มช่วงเวลา vào `self.steps` และเพิ่มชื่อ vào `self.steps_names`
        *   ถ้าเป็น `OPERATOR`, `PYTHON` (CPU op): จะตรวจสอบว่าเป็น communication op หรือไม่ ถ้าใช่ เพิ่ม vào `ProfileRole.Communication` ถ้าไม่ใช่ เพิ่ม vào `ProfileRole.CpuOp`
    *   **ผลลัพธ์**: `self.role_ranges` และ `self.steps` จะค่อยๆ ถูกเติมเต็มด้วยข้อมูลช่วงเวลาของ event ทั้งหมด

### `class EventParser(NodeParserMixin, StepParser)`

*   **หน้าที่**: เป็นคลาสหลักที่รวมการทำงานของ `NodeParserMixin` และ `StepParser` เข้าด้วยกัน และควบคุม flow การทำงานทั้งหมด
*   **โค้ดและคำอธิบายเชิงลึก**:
    ```python
    class EventParser(NodeParserMixin, StepParser):
        def __init__(self):
            super().__init__()
            self.comm_node_list: Dict[CommunicationNode] = None

        def parse(self, events: Iterable[BaseEvent], fwd_bwd_map: Dict[int, int]) -> Dict[int, List[OperatorNode]]:
            # 1. Parse Nodes
            with utils.timing('EventParser: parse nodes'):
                tid2list, tid2zero_rt_list, staled_device_nodes, pl_tid2list = self.parse_nodes(events)

            # 2. Build Op Tree
            with utils.timing('EventParser: build operator tree'):
                builder = OpTreeBuilder()
                tid2tree = builder.build_tree(tid2list, tid2zero_rt_list, staled_device_nodes, fwd_bwd_map=fwd_bwd_map)
                pl_tid2tree = builder.build_tree(pl_tid2list, {}, [], {})

            # 3. Parse Steps
            with utils.timing('EventParser: parse steps times'):
                self.parse_steps(events, self.communication_data)
                # ...

            # 4. Update Step Durations with Device Times
            self.update_device_steps(self.runtime_node_list)

            # 5. Generate and finalize communication nodes
            self.comm_node_list = generate_communication_nodes(self.communication_data, self.steps, self.steps_names)
            return tid2tree, pl_tid2tree
    ```
*   **ลำดับการทำงานในเมธอด `parse`**:
    1.  **`self.parse_nodes(events)`**: เรียกใช้เมธอดจาก `NodeParserMixin` เพื่อแปลง event list เป็น `tid2list` (รายการ OperatorNode ที่ยังไม่มีโครงสร้างพ่อ-ลูก)
    2.  **`builder = OpTreeBuilder()`**: สร้าง instance ของ `OpTreeBuilder` (จาก `op_tree.py`)
    3.  **`tid2tree = builder.build_tree(...)`**: **ขั้นตอนสำคัญ** - ส่ง `tid2list` และ `fwd_bwd_map` ที่ได้รับมา ไปให้ `OpTreeBuilder` เพื่อสร้าง Op Tree ที่สมบูรณ์ (มีโครงสร้างพ่อ-ลูก และมี `BackwardNode`)
    4.  **`self.parse_steps(...)`**: เรียกใช้เมธอดจาก `StepParser` เพื่อสร้าง `self.steps` และ `self.role_ranges`
    5.  **`self.update_device_steps(...)`**: ปรับปรุงเวลาเริ่มต้น-สิ้นสุดของแต่ละ step โดยพิจารณาถึงเวลาของ kernel ที่ทำงานบน device ด้วย
    6.  **`self.comm_node_list = ...`**: เรียกฟังก์ชันจาก `communication.py` เพื่อจัดระเบียบ `CommunicationNode` ที่ได้จาก `NodeParserMixin`
    7.  **`return tid2tree, pl_tid2tree`**: คืนค่า Op Tree ที่สร้างเสร็จแล้วให้กับ `data.py` เพื่อนำไปประมวลผลต่อใน parser อื่นๆ

---
หวังว่าคำอธิบายนี้จะช่วยให้เข้าใจบทบาทและการทำงานของ `event_parser.py` ในฐานะตัวประมวลผล Event หลักของ profiler ได้อย่างละเอียดและชัดเจนครับ
