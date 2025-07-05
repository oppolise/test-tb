# การทำงานของ Torch TB Profiler Plugin

เอกสารนี้อธิบายการทำงานภายในของ Torch TB Profiler Plugin โดยเน้นที่การดึงข้อมูล trace, การแยกแยะระหว่าง computation และ communication, และการทำงานร่วมกันของไฟล์ต่างๆ ในโฟลเดอร์ `tb_plugin/torch_tb_profiler/profiler`

## 1. การดึงข้อมูล Trace และการแยกแยะ Computation/Communication

การดึงข้อมูล trace และการแยกแยะระหว่าง "computation" (การคำนวณ) และ "communication" (การสื่อสารข้อมูล) เป็นหัวใจสำคัญของ profiler เพื่อให้เข้าใจว่าเวลาส่วนใหญ่ถูกใช้ไปกับส่วนใดของการทำงาน

**ขั้นตอนหลัก:**

1.  **การโหลดข้อมูล Trace (`loader.py` -> `RunLoader`):**
    *   Profiler เริ่มจากการโหลดไฟล์ trace ที่ถูกสร้างโดย PyTorch Profiler (ปกติจะเป็นไฟล์ JSON)
    *   สำหรับ distributed training ที่มีหลาย worker หรือหลาย span (ช่วงเวลา), `RunLoader` จะจัดการอ่านไฟล์ของแต่ละส่วนและประมวลผลแบบขนาน (ใช้ `multiprocessing`)

2.  **การสร้าง Event Objects (`data.py` -> `RunProfileData` -> `trace.py`):**
    *   ข้อมูลดิบจากไฟล์ trace (JSON `traceEvents`) จะถูกแปลงเป็น event object ประเภทต่างๆ ที่นิยามใน `trace.py` (เช่น `OperatorEvent`, `KernelEvent`, `RuntimeEvent`)
    *   `trace.py` มีการกำหนด `EventTypes` และ `EventTypeMap` เพื่อแมพ category ของ event จาก JSON ไปเป็นประเภทภายในของ profiler
    *   มีการระบุ `NcclOpNameSet` และ `GlooOpNameSet` ซึ่งเป็น list ของชื่อ operation ที่เกี่ยวข้องกับการสื่อสารข้อมูลแบบ distributed (เช่น `nccl:all_reduce`, `gloo:broadcast`)

3.  **การประมวลผล Event และสร้าง Node (`event_parser.py` -> `EventParser`, `NodeParserMixin`):**
    *   `EventParser` (และ `NodeParserMixin` ที่มันสืบทอดมา) จะวนลูปอ่าน event แต่ละตัวที่สร้างขึ้นในขั้นตอนก่อนหน้า
    *   **การระบุ Communication:**
        *   หาก `event.name` (ชื่อของ operation) ตรงกับชื่อใน `NcclOpNameSet` หรือ `GlooOpNameSet`, event นั้นจะถูกระบุว่าเป็น communication operation
        *   จะมีการสร้าง `CommunicationNode` (นิยามใน `node.py`) เพื่อเก็บข้อมูลของ communication operation นี้ และเก็บไว้ใน `EventParser.communication_data`
    *   **การระบุ Computation:**
        *   Event ที่มีประเภทเป็น `EventTypes.KERNEL` (ทำงานบน GPU) และ *ไม่เกี่ยวข้อง* โดยตรงกับ communication operation (คือ `external_id` ของ kernel ไม่ได้อยู่ใน `communication_data`) จะถูกพิจารณาว่าเป็น GPU computation
        *   Event ที่มีประเภทเป็น `EventTypes.OPERATOR` หรือ `EventTypes.PYTHON` (ทำงานบน CPU) และ *ไม่ใช่* communication operation จะถูกพิจารณาว่าเป็น CPU computation

4.  **การจัดกลุ่มตามบทบาท (Role) ในแต่ละ Step (`event_parser.py` -> `StepParser`):**
    *   `StepParser` จะนำ event ที่ประมวลผลแล้ว มาจัดกลุ่มตามบทบาท (Role) ภายในแต่ละ "step" ของ profiler (เช่น `ProfilerStep#1`, `ProfilerStep#2`)
    *   บทบาทเหล่านี้ถูกนิยามใน `ProfileRole` (เช่น `Kernel`, `Memcpy`, `Communication`, `Runtime`, `CpuOp`)
    *   Kernel event ที่ถูกเรียกโดย communication operation (ตรวจสอบผ่าน `external_id` ที่เชื่อมโยงกับ `communication_data`) จะถูกนับรวมใน `ProfileRole.Communication`
    *   Kernel event อื่นๆ จะถูกนับรวมใน `ProfileRole.Kernel` (GPU computation)
    *   CPU operation ที่ไม่ใช่ communication จะถูกนับรวมใน `ProfileRole.CpuOp` (CPU computation)

**สรุป:** การแยกแยะ computation และ communication อาศัย **ชื่อของ operation** (สำหรับ communication primitives เช่น NCCL, Gloo) และ **ประเภทของ event** (เช่น KERNEL สำหรับ GPU computation, OPERATOR สำหรับ CPU computation) เป็นหลัก โดยมี `event_parser.py` เป็นศูนย์กลางในการทำหน้าที่นี้

## 2. การทำงานของไฟล์ต่างๆ ใน `tb_plugin/torch_tb_profiler/profiler`

โฟลเดอร์นี้เป็นหัวใจหลักของ plugin ประกอบด้วยไฟล์ต่างๆ ที่ทำงานร่วมกันเพื่อประมวลผลข้อมูล profiling และสร้างผลลัพธ์สำหรับแสดงผล

**ลำดับการทำงานและปฏิสัมพันธ์ของไฟล์หลัก:**

1.  **`loader.py` (`RunLoader`):**
    *   **หน้าที่:** โหลดข้อมูล trace จากไฟล์ (รองรับ multi-worker/span)
    *   **การทำงาน:** อ่านไฟล์ JSON, เรียก `RunProfileData.parse()` สำหรับแต่ละไฟล์, และรวมผลลัพธ์
    *   **เรียกใช้:** `data.py` (`RunProfileData`), `run_generator.py` (`RunGenerator`, `DistributedRunGenerator`)

2.  **`data.py` (`RunProfileData`, `DistributedRunProfileData`):**
    *   **หน้าที่:** เป็น class กลางสำหรับเก็บข้อมูลโปรไฟล์ของ worker/span หนึ่งๆ และเรียก parser ต่างๆ มาประมวลผล
    *   **`RunProfileData.process()`:**
        *   **`trace.py`:** สร้าง event objects จาก JSON
        *   **`event_parser.py` (`EventParser`):**
            *   ใช้ `node.py` สร้าง `OperatorNode` เริ่มต้น
            *   แยกแยะ computation/communication, จัดกลุ่ม event ตาม `ProfileRole`
            *   เรียก `op_tree.py` (`OpTreeBuilder`) เพื่อสร้าง op tree ที่สมบูรณ์ (รวม backward pass correlation)
            *   เรียก `communication.py` (`generate_communication_nodes`) เพื่อจัดการ communication node
        *   **`op_agg.py` (`ModuleAggregator`):** รวม (aggregate) ข้อมูล operator และ kernel จาก op tree เพื่อสร้างมุมมองสรุป (เช่น op list by name, kernel list by op)
        *   **`overall_parser.py` (`OverallParser`):** คำนวณค่าใช้จ่าย (cost) โดยรวมของแต่ละ step และค่าเฉลี่ย แบ่งตาม `ProfileRole` และคำนวณ communication overlap
        *   **`gpu_metrics_parser.py` (`GPUMetricsParser`):** ประมวลผล `KernelEvent` เพื่อคำนวณ GPU utilization, SM efficiency, occupancy
        *   **`kernel_parser.py` (`KernelParser`):** รวบรวมสถิติของ kernel (จำนวนครั้ง, เวลารวม/เฉลี่ย, การใช้ Tensor Cores)
        *   **`tensor_cores_parser.py` (`TensorCoresParser`):** วิเคราะห์การใช้งาน Tensor Cores ในระดับ kernel และ operator
        *   **`memory_parser.py` (`MemoryParser`):** วิเคราะห์ `MemoryEvent` เพื่อติดตามการใช้ memory, หา peak memory, และเชื่อมโยงกับ operator
    *   **`RunProfileData.analyze()`:** สร้างคำแนะนำ (recommendations) จากผลการวิเคราะห์
    *   **เรียกใช้โดย:** `loader.py`

3.  **`node.py`:**
    *   **หน้าที่:** นิยามโครงสร้างข้อมูลหลักของ profiler ในรูปแบบของ Node ต่างๆ เช่น `BaseNode`, `OperatorNode`, `DeviceNode` (Kernel, Memcpy), `RuntimeNode`, `CommunicationNode`
    *   `OperatorNode` มีเมธอด `fill_stats()` ที่สำคัญในการคำนวณค่า duration ต่างๆ (self/total host/device time) แบบ recursive
    *   **ถูกใช้โดย:** `event_parser.py` (ในการสร้าง op tree), `op_agg.py` (ในการรวมข้อมูล)

4.  **`event_parser.py` (`EventParser`, `NodeParserMixin`, `StepParser`):**
    *   **หน้าที่:** เป็น parser หลักที่ประมวลผล event list, สร้าง op tree เบื้องต้น, จัดประเภท event, และแบ่งการทำงานออกเป็น step และ role
    *   **`NodeParserMixin._parse_node()`:** สร้าง Node object (จาก `node.py`) สำหรับแต่ละ event และระบุ communication node
    *   **`StepParser._parse_step()`:** แบ่งช่วงเวลาการทำงานเป็น step และจัดกลุ่ม event ตาม `ProfileRole` (เช่น Kernel, Communication, CpuOp)
    *   **เรียกใช้:** `op_tree.py`, `communication.py`
    *   **ถูกเรียกโดย:** `data.py`

5.  **`op_tree.py` (`OpTreeBuilder`):**
    *   **หน้าที่:** สร้างโครงสร้างต้นไม้ของ operator (op tree) ที่สมบูรณ์จาก `OperatorNode` ที่ได้จาก `EventParser`
    *   จัดการการสร้างความสัมพันธ์แบบพ่อ-ลูก (parent-child)
    *   รวม node ที่ซ้ำซ้อน (duplicate nodes)
    *   **ที่สำคัญ:** ทำการเชื่อมโยง forward pass กับ backward pass (Backward Correlation) โดยใช้ `fwd_bwd_map` จาก `trace.py` เพื่อสร้าง `BackwardNode` และแทรกเข้าไปใน op tree
    *   **ถูกเรียกโดย:** `event_parser.py`

6.  **Parser เฉพาะทาง:**
    *   **`gpu_metrics_parser.py`:** คำนวณ GPU utilization, SM efficiency, occupancy
    *   **`kernel_parser.py`:** สรุปสถิติ kernel (calls, duration, TC usage)
    *   **`memory_parser.py`:** วิเคราะห์ memory usage, peak memory
    *   **`overall_parser.py`:** คำนวณ step costs, average costs, communication overlap
    *   **`tensor_cores_parser.py`:** วิเคราะห์การใช้ Tensor Cores
    *   **ถูกเรียกโดย:** `data.py` (`RunProfileData.process()`)

7.  **`communication.py`:**
    *   **หน้าที่:** ฟังก์ชันเสริมสำหรับประมวลผล `CommunicationNode`
    *   `generate_communication_nodes()`: ระบุ step ของแต่ละ communication op
    *   `analyze_communication_nodes()`: คำนวณสถิติ communication (latency, data transfer time) สำหรับ distributed view
    *   **ถูกเรียกโดย:** `event_parser.py`, `data.py` (`DistributedRunProfileData`)

8.  **`op_agg.py` (`ModuleAggregator`, `OperatorAgg`, `KernelAggByNameOp`):**
    *   **หน้าที่:** รวม (aggregate) ข้อมูลจาก op tree เพื่อสร้างข้อมูลสรุปสำหรับแสดงผลใน Operator view และ Kernel view
    *   `aggregate_ops()`: จัดกลุ่ม operator ตาม key (เช่น ชื่อ, ชื่อ+input shape) และคำนวณค่าสรุป
    *   `aggregate_kernels()`: จัดกลุ่ม kernel และคำนวณค่าสรุป
    *   **ถูกเรียกโดย:** `data.py` (`RunProfileData.process()`)

9.  **`run_generator.py` (`RunGenerator`, `DistributedRunGenerator`):**
    *   **หน้าที่:** เป็นขั้นตอนสุดท้ายในการเตรียมข้อมูลสำหรับ TensorBoard UI
    *   รับ `RunProfileData` (หรือ list ของ `DistributedRunProfileData`) ที่ผ่านการประมวลผลทั้งหมดแล้ว
    *   สร้าง object `RunProfile` (หรือ `DistributedRunProfile`) ซึ่งมีข้อมูลที่จัดรูปแบบพร้อมสำหรับแสดงผลใน view ต่างๆ ของ TensorBoard (Overall, Operator, Kernel, Trace, Memory, Distributed)
    *   **ถูกเรียกโดย:** `loader.py`

**โฟลว์ข้อมูลโดยสรุป:**

`Trace File (JSON)` -> `loader.py` -> `data.py (RunProfileData)` -> (เรียก parser ต่างๆ: `event_parser.py`, `op_tree.py`, `gpu_metrics_parser.py`, etc. และ `op_agg.py`) -> `run_generator.py` -> `RunProfile Object (สำหรับ UI)`

ไฟล์ย่อยอื่นๆ เช่น `range_utils.py` (จัดการช่วงเวลา), `tensor_core.py` (list ของ op ที่ Tensor Core eligible), `module_op.py` (จัดการ module view) ก็มีบทบาทสนับสนุนการทำงานของไฟล์หลักเหล่านี้

หวังว่าเอกสารนี้จะเป็นประโยชน์ในการทำความเข้าใจการทำงานของ Torch TB Profiler Plugin ครับ
