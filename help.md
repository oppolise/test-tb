# ความเข้าใจเกี่ยวกับ PyTorch Profiler Plugin และการดึงข้อมูล Computation/Communication

เอกสารนี้สรุปการทำงานของ plugin profiler ในส่วน `tb_plugin/torch_tb_profiler/profiler` โดยเน้นไปที่วิธีการดึงข้อมูล computation และ communication รวมถึงความสัมพันธ์และการทำงานของไฟล์ต่างๆ ภายในโฟลเดอร์นี้

## 1. การดึงข้อมูล Computation และ Communication

Profiler Plugin แยกแยะและดึงข้อมูลการคำนวณ (Computation) และการสื่อสาร (Communication) โดยอาศัยการวิเคราะห์ประเภทและชื่อของ event ที่ถูกบันทึกระหว่างการ profiling

**การระบุ Communication Events:**

หัวใจหลักของการระบุ communication events อยู่ที่ไฟล์ `profiler/trace.py` และ `profiler/event_parser.py`:

1.  **`trace.py`**:
    *   กำหนด `NcclOpNameSet` และ `GlooOpNameSet` ซึ่งเป็นลิสต์ของชื่อ operation ที่เกี่ยวข้องกับการสื่อสารแบบ distributed (เช่น `nccl:all_reduce`, `gloo:broadcast`)
    *   `NcclOpNameSet = ['nccl:broadcast', 'nccl:reduce', 'nccl:all_reduce', 'nccl:all_gather', 'nccl:reduce_scatter']`
    *   `GlooOpNameSet = ['gloo:broadcast', 'gloo:reduce', 'gloo:all_reduce', 'gloo:all_gather', 'gloo:reduce_scatter']`

2.  **`event_parser.py` (ภายในคลาส `NodeParserMixin`, เมธอด `_parse_node`)**:
    *   เมื่อมีการประมวลผล `OperatorEvent` (หรือ `DurationEvent` ที่มี category เป็น `operator`), โค้ดจะตรวจสอบว่า `event.name` (ชื่อของ operator) อยู่ใน `NcclOpNameSet` หรือ `GlooOpNameSet` หรือไม่
        ```python
        if event.name in NcclOpNameSet or event.name in GlooOpNameSet:
            comm_node = CommunicationNode.create(event)
            # ...เก็บ comm_node ใน self.communication_data โดยใช้ external_id ของ op เป็น key
        ```
    *   หากชื่อตรงกัน จะมีการสร้าง `CommunicationNode` (นิยามใน `profiler/node.py`) ขึ้นมาเพื่อเก็บข้อมูลของ communication operation นี้โดยเฉพาะ
    *   Kernel events ที่เกี่ยวข้องกับ communication operation นี้ (เชื่อมโยงผ่าน `external_id`) จะถูกนำไปอัปเดตข้อมูลใน `CommunicationNode` ที่เกี่ยวข้อง (ในเมธอด `_update_communication_node`) เช่น การเพิ่มช่วงเวลาของ kernel (`kernel_ranges`) และเวลารวม (`total_time`)

3.  **`event_parser.py` (ภายในคลาส `StepParser`, เมธอด `_parse_step`)**:
    *   ในช่วงการแบ่งเวลาการทำงานออกเป็น step ต่างๆ และคำนวณสัดส่วนเวลาของแต่ละประเภท (role) ใน step นั้น:
        *   ถ้า `KernelEvent` มี `external_id` ที่ตรงกับ `CommunicationNode` ที่ถูกระบุก่อนหน้า, เวลาของ kernel นั้นจะถูกนับเข้า `ProfileRole.Communication`.
        *   ถ้า `OperatorEvent` (บน CPU) มีชื่ออยู่ใน `NcclOpNameSet` หรือ `GlooOpNameSet`, เวลาของ operator นั้นก็จะถูกนับเข้า `ProfileRole.Communication` เช่นกัน.

**การระบุ Computation Events:**

การระบุ computation events ไม่ได้ทำโดยตรงผ่าน keyword เหมือน communication แต่เป็นการจัดประเภทเวลาที่เหลือหลังจากที่ communication, memory operations, runtime, และ dataloader ถูกแยกออกไปแล้ว:

1.  **`event_parser.py` (ภายในคลาส `StepParser`, เมธอด `_parse_step`)**:
    *   `KernelEvent` ที่ **ไม่ใช่** ส่วนหนึ่งของ communication operation (คือ `external_id` ไม่ได้อยู่ใน `communication_data`) จะถูกจัดประเภทเป็น `ProfileRole.Kernel` นี่คือส่วนหลักของ "Device Computation"
    *   `OperatorEvent` (บน CPU) ที่ **ไม่ใช่** communication operation และไม่ใช่ dataloader operation จะถูกจัดประเภทเป็น `ProfileRole.CpuOp` นี่คือส่วนหลักของ "Host Computation"

**สรุปกระบวนการ:**

*   Communication events ถูกดึงออกมาโดยการจับคู่ชื่อ operator กับลิสต์ที่กำหนดไว้ล่วงหน้า และ kernel ที่เกี่ยวข้องจะถูกเชื่อมโยง.
*   Computation events คือเวลาของ kernel และ CPU operator ที่เหลือหลังจากหักส่วนที่เป็น communication และส่วนอื่นๆ (memory, runtime, dataloader) ออกไปแล้ว.

ข้อมูล `ProfileRole` ที่ถูกแบ่งประเภทนี้ (รวมถึง `ProfileRole.Communication`, `ProfileRole.Kernel`, `ProfileRole.CpuOp`) จะถูกนำไปใช้ต่อใน `OverallParser` เพื่อคำนวณสถิติภาพรวมของเวลาที่ใช้ในแต่ละ step และการซ้อนทับกัน (overlap) ระหว่าง computation และ communication.

## 2. การทำงานของไฟล์ใน `profiler` โฟลเดอร์และความสัมพันธ์

โฟลเดอร์ `tb_plugin/torch_tb_profiler/profiler` ประกอบด้วยไฟล์หลายตัวที่ทำงานร่วมกันเพื่อประมวลผลข้อมูลจาก profiler และเตรียมข้อมูลสำหรับการแสดงผลใน TensorBoard UI

**การไหลของข้อมูลหลัก (Simplified):**

`loader.py` -> `data.py` (เรียก `event_parser.py`) -> `event_parser.py` (เรียก `trace.py`, `node.py`, `op_tree.py`, `communication.py`, `range_utils.py`) -> `data.py` (เรียก parser อื่นๆ เช่น `overall_parser.py`, `gpu_metrics_parser.py`) -> `run_generator.py` (สร้างผลลัพธ์สำหรับ UI)

**คำอธิบายไฟล์หลักและหน้าที่:**

*   **`trace.py`**:
    *   **หน้าที่**: กำหนดประเภทของ event ต่างๆ (`EventTypes`), โครงสร้างข้อมูลพื้นฐานสำหรับแต่ละ event (`BaseEvent`, `DurationEvent`, `KernelEvent`, `OperatorEvent`, `MemoryEvent` ฯลฯ) และฟังก์ชันสำหรับสร้าง event object (`create_event`) จากข้อมูลดิบใน trace file.
    *   **จุดสำคัญ**: นิยาม `NcclOpNameSet` และ `GlooOpNameSet` ที่ใช้ระบุ communication operations.
    *   **ถูกเรียกใช้โดย**: `data.py` (เพื่อสร้าง event objects), `event_parser.py` (เพื่ออ้างอิง `EventTypes` และชื่อ communication ops).

*   **`event_parser.py`**:
    *   **หน้าที่**: เป็นหัวใจหลักในการประมวลผล stream ของ event ทั้งหมด. ประกอบด้วย `NodeParserMixin` (สำหรับสร้างโครงสร้าง node ต่างๆ เช่น operator, runtime, device, communication node) และ `StepParser` (สำหรับแบ่งการทำงานเป็น step และคำนวณสัดส่วนเวลาของแต่ละประเภทกิจกรรมใน step).
    *   **การทำงาน**:
        *   สร้าง operator tree (`OpTreeBuilder` จาก `op_tree.py`).
        *   **ระบุและสร้าง `CommunicationNode`** เมื่อเจอ operator ที่ชื่อตรงกับ `NcclOpNameSet` หรือ `GlooOpNameSet`.
        *   เชื่อมโยง kernel events เข้ากับ `CommunicationNode` ที่เกี่ยวข้อง.
        *   จัดประเภทช่วงเวลาของ event ต่างๆ เข้า `ProfileRole` (เช่น `Communication`, `Kernel`, `CpuOp`, `DataLoader`, `Runtime`, `Memcpy`, `Memset`, `Other`).
        *   เรียก `communication.generate_communication_nodes()` เพื่อประมวลผล `CommunicationNode` ที่รวบรวมได้.
    *   **ถูกเรียกใช้โดย**: `data.py` (`RunProfileData.process()`).
    *   **เรียกใช้**: `trace.py`, `node.py`, `op_tree.py`, `communication.py`, `range_utils.py`.

*   **`node.py`**:
    *   **หน้าที่**: นิยามคลาสสำหรับ Node ประเภทต่างๆ ที่ใช้ในการสร้าง operator tree และเก็บข้อมูลเฉพาะของแต่ละ node (เช่น `OperatorNode`, `DeviceNode`, `RuntimeNode`, `CommunicationNode`, `ProfilerStepNode`, `ModuleNode`).
    *   **ถูกเรียกใช้โดย**: `event_parser.py` (ในการสร้าง node), `communication.py` (อ้างอิง `CommunicationNode`), `op_tree.py`, `op_agg.py`, `diffrun/operator.py`, `diffrun/tree.py`.

*   **`communication.py`**:
    *   **หน้าที่**: จัดการและวิเคราะห์ `CommunicationNode` โดยเฉพาะ.
    *   `generate_communication_nodes()`: รับ `communication_data` ดิบจาก `EventParser`, จัดเรียง, และกำหนด `step_name` ให้แต่ละ `CommunicationNode`.
    *   `analyze_communication_nodes()`: คำนวณสถิติ communication โดยละเอียด (ต่อ step, ต่อชื่อ op) เช่น latency, data transfer time, โดยใช้ `kernel_ranges` และ `real_time_ranges` จาก `CommunicationNode`.
    *   **ถูกเรียกใช้โดย**: `event_parser.py` (เรียก `generate_communication_nodes`), `data.py` (`DistributedRunProfileData` เรียก `analyze_communication_nodes`).
    *   **เรียกใช้**: `node.py` (`CommunicationNode`), `range_utils.py`.

*   **`data.py`**:
    *   **`RunProfileData`**: คลาสหลักที่ encapsulate ข้อมูลทั้งหมดของ profiler run หนึ่งๆ (สำหรับ worker หนึ่ง).
        *   อ่าน trace file (JSON), เรียก `trace.create_event` เพื่อสร้าง event objects.
        *   เรียก `EventParser.parse()` เพื่อประมวลผล event ทั้งหมด และรับผลลัพธ์เช่น operator tree (`tid2tree`), `comm_node_list`, `role_ranges`, `steps_names`, `used_devices`.
        *   เรียก parser เฉพาะทางอื่นๆ (`OverallParser`, `GPUMetricsParser`, `TensorCoresParser`, `KernelParser`, `MemoryParser`) เพื่อประมวลผลข้อมูลในแต่ละด้าน.
        *   มีเมธอด `analyze()` เพื่อสร้างคำแนะนำ (recommendations).
    *   **`DistributedRunProfileData`**: เก็บข้อมูลที่เกี่ยวข้องกับ distributed view จาก `RunProfileData` และเรียก `communication.analyze_communication_nodes()` เพื่อคำนวณสถิติ communication ที่อาจมีการปรับแก้ `real_time_ranges` แล้ว.
    *   **ถูกเรียกใช้โดย**: `loader.py`.
    *   **เรียกใช้**: `trace.py`, `event_parser.py`, `communication.py` (ทางอ้อมผ่าน `DistributedRunProfileData`), และ parser เฉพาะทางทั้งหมด.

*   **`loader.py`**:
    *   **`RunLoader`**: รับผิดชอบในการโหลดข้อมูล profiler จาก run directory ซึ่งอาจมีไฟล์ trace จากหลาย worker และหลาย span.
    *   ใช้ multiprocessing เพื่อประมวลผลแต่ละไฟล์ trace แบบขนาน.
    *   แต่ละ process จะเรียก `RunProfileData.parse()` เพื่อประมวลผลไฟล์ trace.
    *   สำหรับ distributed run, จะมีการปรับแก้ `real_time_ranges` ใน `CommunicationNode` ของแต่ละ worker เพื่อสะท้อนเวลา communication ที่ "แท้จริง" ก่อนที่จะเรียก `communication_parse()` บน `DistributedRunProfileData`.
    *   **ถูกเรียกใช้โดย**: ส่วนหลักของ plugin เมื่อต้องการโหลดข้อมูล run.
    *   **เรียกใช้**: `data.py` (`RunProfileData`, `DistributedRunProfileData`), `run_generator.py`.

*   **`run_generator.py`**:
    *   **`RunGenerator`**: รับ `RunProfileData` ที่ประมวลผลสมบูรณ์แล้ว มาจัดรูปแบบข้อมูลสำหรับแสดงผลใน view ต่างๆ ของ TensorBoard UI (Overall, Operator, Kernel, Trace, Memory, Module).
    *   **`DistributedRunGenerator`**: รับ list ของ `DistributedRunProfileData` มาจัดรูปแบบข้อมูลสำหรับ distributed view (Computation/Communication Overview, Synchronizing/Communication Overview, Communication Operations Stats).
    *   **ถูกเรียกใช้โดย**: `loader.py`.
    *   **เรียกใช้**: `module_op.py`.

**Parser เฉพาะทาง:**

*   **`overall_parser.py` (`OverallParser`)**:
    *   **หน้าที่**: คำนวณสถิติภาพรวมของเวลาที่ใช้ในแต่ละ step และค่าเฉลี่ย โดยแบ่งตาม `ProfileRole` (ที่ได้จาก `EventParser`). คำนวณส่วนที่ computation (Kernel/CpuOp) และ communication เกิดขึ้นพร้อมกัน (overlap).
    *   **เกี่ยวข้องกับ Comp/Comm**: ใช้ข้อมูล `role_ranges` ที่แยกประเภทแล้ว (รวมถึง `Communication`, `Kernel`, `CpuOp`) ในการคำนวณสัดส่วนเวลาและ overlap.
*   **`gpu_metrics_parser.py` (`GPUMetricsParser`)**:
    *   **หน้าที่**: ประมวลผล `KernelEvent` เพื่อคำนวณ GPU Utilization, Estimated SM Efficiency, และ Average Occupancy.
    *   **เกี่ยวข้องกับ Comp/Comm**: วิเคราะห์ `KernelEvent` ทั้งหมด ไม่ได้แยกแยะโดยตรงว่า kernel มาจาก computation หรือ communication แต่ผลลัพธ์จะสะท้อนการทำงานของ kernel ทั้งหมดบน GPU.
*   **`kernel_parser.py` (`KernelParser`)**:
    *   **หน้าที่**: สร้างสถิติของแต่ละ kernel (จำนวนครั้ง, เวลารวม/เฉลี่ย/min/max) และคำนวณสัดส่วนเวลา kernel ที่ใช้ Tensor Cores.
    *   **เกี่ยวข้องกับ Comp/Comm**: เหมือน `GPUMetricsParser`, วิเคราะห์ `KernelEvent` ทั้งหมด. ถ้า communication op มี kernel, ก็จะถูกรวมในสถิติ.
*   **`memory_parser.py` (`MemoryParser`)**:
    *   **หน้าที่**: วิเคราะห์ `MemoryEvent`, เชื่อมโยงการจัดสรร/คืนหน่วยความจำกับ operator ที่เกี่ยวข้อง, และคำนวณสถิติการใช้หน่วยความจำ.
    *   **เกี่ยวข้องกับ Comp/Comm**: ทางอ้อม; ถ้า computation หรือ communication มีการใช้ memory ที่ถูก track, parser นี้จะพยายาม map เข้ากับ operator นั้น.
*   **`tensor_cores_parser.py` (`TensorCoresParser`)**:
    *   **หน้าที่**: คำนวณสัดส่วนเวลาที่ kernel ใช้ Tensor Cores (TC) ต่อ GPU และสัดส่วนเวลา kernel ที่ถูกเรียกโดย operator ที่ TC-eligible.
    *   **เกี่ยวข้องกับ Comp/Comm**: เน้นคุณสมบัติของ kernel/operator ที่เกี่ยวกับ TC (computation). ถ้า communication op ใช้ kernel ที่ TC-eligible หรือใช้ TC, ก็จะถูกนับรวม.

**ไฟล์ Utility:**

*   **`op_tree.py` (`OpTreeBuilder`)**: สร้าง operator hierarchy จาก list ของ operator nodes. ถูกใช้โดย `EventParser`.
*   **`op_agg.py` (`ModuleAggregator`, `OperatorAgg`)**: รวมข้อมูล operator สำหรับแสดงผลใน views. ถูกใช้โดย `RunProfileData`.
*   **`range_utils.py`**: ฟังก์ชันช่วยในการจัดการกับ list ของช่วงเวลา (เช่น merge, intersection, sum). ถูกใช้โดย `EventParser`, `communication.py`, `overall_parser.py`, `gpu_metrics_parser.py`.
*   **`module_op.py`**: ฟังก์ชันสำหรับสร้างข้อมูลใน Module View / Lightning View. ถูกใช้โดย `RunGenerator`.
*   **`tensor_core.py` (`TC_Allowlist`)**: ลิสต์ของชื่อ kernel ที่ถือว่าใช้ Tensor Cores. ถูกใช้โดย `KernelParser` และ `KernelEvent`.

**โฟลเดอร์ `diffrun` (`contract.py`, `operator.py`, `tree.py`):**

*   **หน้าที่**: เปรียบเทียบโครงสร้างและประสิทธิภาพ (host/device duration) ของ operator tree จาก profiler run สองครั้ง.
*   **เกี่ยวข้องกับ Comp/Comm**:
    *   ใช้ `OperatorNode.device_duration` (สะท้อน device computation time) ในการเปรียบเทียบ.
    *   Communication ops ที่เป็น `OperatorNode` จะถูกรวมในการเปรียบเทียบด้วย.
    *   ไม่ได้วิเคราะห์ communication metrics โดยละเอียด (เช่น overlap, bandwidth) เหมือนส่วนอื่น.

เอกสารนี้ควรให้ภาพรวมที่ดีเกี่ยวกับวิธีการทำงานของ profiler plugin ในส่วนที่สนใจแล้วครับ
