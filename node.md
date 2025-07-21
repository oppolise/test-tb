# อธิบายโค้ด `node.py` เชิงลึก

เอกสารนี้จะอธิบายการทำงานของไฟล์ `tb_plugin/torch_tb_profiler/profiler/node.py` อย่างละเอียด เพื่อให้เข้าใจถึงวิธีการที่ไฟล์นี้ทำหน้าที่เป็น **"พิมพ์เขียว" (Blueprint) สำหรับโครงสร้างข้อมูลทั้งหมด** ที่ใช้ในการสร้าง Op Tree

## ภาพรวมของ `node.py`

ไฟล์ `node.py` คือหัวใจของโครงสร้างข้อมูลใน profiler นี้ มันไม่ได้ทำการประมวลผลที่ซับซ้อน แต่ทำหน้าที่ที่สำคัญอย่างยิ่งคือ **การนิยามคลาส (class) ต่างๆ ที่ใช้แทนองค์ประกอบแต่ละอย่างใน profiling trace** เช่น Operator, Kernel, Runtime call, Communication operation

คลาสเหล่านี้มีลักษณะเป็นลำดับชั้น (hierarchy) โดยมี `BaseNode` เป็นคลาสแม่สูงสุด และคลาสอื่นๆ จะสืบทอด (inherit) คุณสมบัติและเพิ่ม attribute หรือเมธอดที่เฉพาะเจาะจงของตัวเองเข้าไป การมีโครงสร้างคลาสที่ชัดเจนเช่นนี้ ทำให้ส่วนอื่นๆ ของโค้ด (โดยเฉพาะ `event_parser.py` และ `op_tree.py`) สามารถสร้างและจัดการกับ object เหล่านี้ได้อย่างเป็นระบบ

---

## การทำงานของโค้ดแต่ละส่วน

### `class BaseNode(ABC)`

*   **หน้าที่**: เป็น Abstract Base Class (ABC) หรือคลาสแม่แบบที่เป็นรากฐานที่สุดสำหรับ Node ทุกประเภทใน Op Tree กำหนดคุณสมบัติพื้นฐานที่ Node ทุกตัวต้องมี
*   **โค้ด**:
    ```python
    class BaseNode(ABC):
        def __init__(self, name: str, start_time: int, end_time: int, type: str, tid: int,
                     external_id: Optional[int] = None):
            self.name = name
            self.start_time = start_time
            self.end_time = end_time
            self.type = type
            self.tid = tid
            self.external_id = external_id  # For consistency check.
    ```
*   **คำอธิบาย `__init__`**:
    *   `self.name`: (str) ชื่อของ Node เช่น "conv2d", "cudaLaunchKernel"
    *   `self.start_time`: (int) Timestamp ตอนเริ่มต้นทำงาน (หน่วยเป็นไมโครวินาที)
    *   `self.end_time`: (int) Timestamp ตอนสิ้นสุดการทำงาน
    *   `self.type`: (str) ประเภทของ Node ซึ่งจะมาจาก `EventTypes` ใน `trace.py` เช่น "Operator", "Kernel"
    *   `self.tid`: (int) Thread ID ที่ Node นี้ทำงานอยู่
    *   `self.external_id`: (int, optional) ID ที่ใช้เชื่อมโยง Operator (บน CPU) กับ Runtime call (บน CPU) ที่มันเรียก

*   **โค้ด `get_node_argument`**:
    ```python
    @staticmethod
    def get_node_argument(event: DurationEvent):
        kwargs = {}
        kwargs['name'] = event.name
        kwargs['start_time'] = event.ts
        kwargs['end_time'] = event.ts + event.duration
        # ... (และอื่นๆ) ...
        return kwargs
    ```
    *   **หน้าที่**: เป็น static method ที่ช่วยอำนวยความสะดวกในการแปลง `DurationEvent` object (จาก `trace.py`) ให้กลายเป็น dictionary (`kwargs`) ที่มี key ตรงกับ argument ของ `BaseNode.__init__`
    *   **ผลลัพธ์**: ทำให้การสร้าง Node object ในคลาสลูกทำได้ง่ายขึ้นโดยการเรียก `BaseNode.get_node_argument(event)` แล้วส่งผลลัพธ์ต่อไป

*   **โค้ด `duration`**:
    ```python
    @property
    def duration(self) -> int:
        return self.end_time - self.start_time
    ```
    *   **หน้าที่**: เป็น property ที่คำนวณระยะเวลาการทำงานของ Node
    *   **ผลลัพธ์**: สามารถเรียกใช้ `node.duration` เพื่อดูระยะเวลาได้โดยตรง ไม่ต้องคำนวณเองทุกครั้ง

### `class CommunicationNode(BaseNode)`

*   **หน้าที่**: ใช้แทน Communication Operation (เช่น `nccl:all_reduce`)
*   **โค้ด**:
    ```python
    class CommunicationNode(BaseNode):
        def __init__(self, input_shape: List[List[int]], input_type: List[str], **kwargs):
            super().__init__(**kwargs)
            self.input_shape = input_shape
            self.input_type = input_type
            self.kernel_ranges: List[Tuple[int, int]] = []
            self.real_time_ranges: List[Tuple[int, int]] = []
            # ...
    ```
*   **คำอธิบาย**:
    *   สืบทอดจาก `BaseNode` และเพิ่ม attribute ที่เกี่ยวกับ communication:
    *   `self.input_shape`, `self.input_type`: ข้อมูลเกี่ยวกับ tensor ที่ถูกส่ง
    *   `self.kernel_ranges`: รายการช่วงเวลาของ kernel ที่ถูกเรียกโดย communication op นี้
    *   `self.real_time_ranges`: รายการช่วงเวลาที่คำนวณแล้วว่าเป็น "เวลาสื่อสารข้อมูลจริงๆ" (ใช้ใน distributed view)

### `class HostNode(BaseNode)`

*   **หน้าที่**: เป็นคลาสแม่สำหรับ Node ที่ทำงานบน Host (CPU) เช่น Operator และ Runtime call
*   **โค้ด**:
    ```python
    class HostNode(BaseNode):
        def __init__(self, device_duration: int = 0, **kwargs):
            super().__init__(**kwargs)
            self.device_duration = device_duration
    ```
*   **คำอธิบาย**:
    *   เพิ่ม `self.device_duration`: (int) ใช้เก็บ "เวลารวม" ของทุกอย่างที่ทำงานบน Device (GPU) ซึ่งถูกเรียกโดย Host Node นี้ (เช่น เวลารวมของ kernel ทั้งหมดที่ถูก launch โดย operator นี้)

### `class OperatorNode(HostNode)`

*   **หน้าที่**: **เป็นคลาสที่สำคัญที่สุดใน Op Tree** ใช้แทน PyTorch Operator ที่ทำงานบน CPU
*   **โค้ด `__init__`**:
    ```python
    class OperatorNode(HostNode):
        def __init__(self, children=None, runtimes=None, ..., self_host_duration: int = 0, self_device_duration: int = 0, **kwargs):
            super().__init__(**kwargs)
            self.children: List[OperatorNode] = [] if children is None else children
            self.runtimes: List[RuntimeNode] = [] if runtimes is None else runtimes
            self.input_shape = input_shape
            self.callstack = callstack
            self.self_host_duration = self_host_duration
            self.self_device_duration = self_device_duration
            self.tc_eligible = self.name in TC_OP_Allowlist
            self.tc_self_duration = 0
            self.tc_total_duration = 0
    ```
*   **คำอธิบาย `__init__`**:
    *   `self.children`: **สำคัญมาก** - List ที่เก็บ `OperatorNode` ลูก (sub-operators) ที่ถูกเรียกโดย Node นี้ เป็นหัวใจของการสร้างโครงสร้างแบบ tree
    *   `self.runtimes`: List ที่เก็บ `RuntimeNode` (เช่น `cudaLaunchKernel`) ที่ถูกเรียกโดยตรงจาก Node นี้
    *   `self.self_host_duration`: เวลาที่ใช้บน CPU ของ Node นี้ *เท่านั้น* (ไม่รวมเวลาของ children และ runtimes)
    *   `self.self_device_duration`: เวลารวมของ device event ที่ถูกเรียกโดย Node นี้ *เท่านั้น* (ไม่รวมของ children)
    *   `self.tc_...`: attribute ที่ใช้เก็บข้อมูลเกี่ยวกับการใช้งาน Tensor Cores

*   **โค้ด `fill_stats()`**:
    ```python
    def fill_stats(self):
        # ... (sort children and runtimes) ...

        for child in self.children:
            child.fill_stats() # Recursive call
        for rt in self.runtimes:
            rt.fill_stats(self)

        self.self_host_duration = self.end_time - self.start_time
        for child in self.children:
            self.device_duration += child.device_duration
            self.self_host_duration -= (child.end_time - child.start_time)
            # ... (รวมค่า tc_total_duration จาก child) ...
        for rt in self.runtimes:
            self.self_host_duration -= (rt.end_time - rt.start_time)
            self.device_duration += rt.device_duration
            self.self_device_duration += rt.device_duration
            # ... (รวมค่า tc_self_duration และ tc_total_duration จาก runtime) ...
    ```
    *   **หน้าที่**: **เมธอดสำคัญมาก** - ทำหน้าที่คำนวณค่าสถิติเชิงลึก (เช่น `self_host_duration`, `device_duration`) แบบ **recursive** (เรียกตัวเองใน children) จากล่างขึ้นบน (post-order traversal)
    *   **การทำงาน**:
        1.  จัดเรียง `children` และ `runtimes` ตามเวลาเริ่มต้น
        2.  **Recursive Call**: เรียก `child.fill_stats()` สำหรับ child ทุกตัว เพื่อให้ child คำนวณค่าของตัวเองให้เสร็จก่อน
        3.  เรียก `rt.fill_stats(self)` สำหรับ runtime ทุกตัว
        4.  **คำนวณ `self_host_duration`**: เริ่มจากระยะเวลาทั้งหมดของตัวเอง (`self.end_time - self.start_time`) แล้ว **ลบออก** ด้วยระยะเวลาของ `children` และ `runtimes` ทั้งหมด
        5.  **คำนวณ `device_duration` (total)**: เริ่มจากค่าของตัวเอง แล้ว **บวกเพิ่ม** ด้วย `device_duration` ของ `children` และ `runtimes` ทั้งหมด
    *   **ผลลัพธ์**: ทำให้ Node แต่ละตัวใน tree มีค่า `self_..._duration` (เวลาของตัวเองจริงๆ) และค่า `total_..._duration` (เวลารวมของตัวเองและลูกหลานทั้งหมด) ที่ถูกต้อง

*   **คลาสลูกของ `OperatorNode`**:
    *   `ProfilerStepNode`, `ModuleNode`, `BackwardNode`, `DataLoaderNode`, `OptimizerNode`: เป็นคลาสเฉพาะทางที่สืบทอดจาก `OperatorNode` เพื่อใช้แทน operation ประเภทพิเศษเหล่านั้น บางคลาสอาจมีการ override เมธอด `fill_stats` หรือ `create` เล็กน้อยเพื่อให้ทำงานได้ถูกต้องตามประเภทของมัน เช่น `BackwardNode.fill_stats` จะคำนวณ `start_time` และ `end_time` ของตัวเองจาก children ของมัน

### `class RuntimeNode(HostNode)`

*   **หน้าที่**: ใช้แทน Runtime API call (เช่น `cudaLaunchKernel`, `cudaMemcpy`)
*   **โค้ด**:
    ```python
    class RuntimeNode(HostNode):
        def __init__(self, device_nodes: Optional[List['DeviceNode']] = None, **kwargs):
            super().__init__(**kwargs)
            self.device_nodes = sorted(device_nodes, ...) if device_nodes else None
            self.tc_duration: int = 0
    ```
*   **คำอธิบาย**:
    *   `self.device_nodes`: **สำคัญ** - List ที่เก็บ `DeviceNode` (เช่น `KernelEvent`, `MemcpyEvent`) ที่ถูกเรียกโดย Runtime call นี้
    *   เมธอด `fill_stats(self, op_node: OperatorNode = None)` ของคลาสนี้จะคำนวณ `device_duration` ของตัวเองโดยการบวกระยะเวลาของ `device_nodes` ทั้งหมดที่มันมี

### `class DeviceNode(BaseNode)`

*   **หน้าที่**: ใช้แทน event ที่ทำงานบน Device (GPU) โดยตรง เช่น Kernel, Memcpy, Memset
*   **โค้ด**:
    ```python
    class DeviceNode(BaseNode):
        def __init__(self, blocks_per_sm: Optional[float] = None, occupancy: int = None, ...):
            super().__init__(**kwargs)
            self.op_tc_eligible = False
            self.op_name = None
            self.blocks_per_sm = blocks_per_sm
            self.occupancy = occupancy
            # ...
            self.tc_used = self.name in TC_Allowlist
    ```
*   **คำอธิบาย**:
    *   เก็บข้อมูลเฉพาะทางของ GPU kernel เช่น `blocks_per_sm`, `occupancy`, `grid`, `block`
    *   `self.op_name`, `self.op_tc_eligible`: จะถูกกำหนดค่าโดย `RuntimeNode.fill_stats` เพื่อให้ `DeviceNode` รู้ว่ามันถูกเรียกโดย Operator ชื่ออะไร และ op นั้น TC eligible หรือไม่
    *   `self.tc_used`: ตรวจสอบว่าชื่อของ kernel นี้อยู่ใน `TC_Allowlist` (รายการของ kernel ที่ใช้ Tensor Cores) หรือไม่

### ฟังก์ชัน Helper

*   **`create_operator_node(event: OperatorEvent)`**:
    *   **หน้าที่**: เป็น factory function ที่ช่วยสร้าง Node ที่เหมาะสมสำหรับ Operator event โดยดูจากชื่อของ event เช่น ถ้าชื่อขึ้นต้นด้วย `enumerate(DataLoader)#` จะสร้าง `DataLoaderNode`
*   **`is_operator_node(node: BaseNode)`**:
    *   **หน้าที่**: ตรวจสอบว่า node ที่ให้มาเป็น "operator ที่น่าสนใจ" หรือไม่ (ไม่ใช่พวก `DataParallel.forward` หรือ `Optimizer.zero_grad` ที่ไม่ต้องการนำมาแสดงในบางมุมมอง)

---
หวังว่าคำอธิบายนี้จะช่วยให้เข้าใจบทบาทและการทำงานของคลาสต่างๆ ใน `node.py` ซึ่งเป็นพิมพ์เขียวของโครงสร้างข้อมูลทั้งหมดใน profiler ได้อย่างละเอียดครับ
