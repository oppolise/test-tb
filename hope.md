# การศึกษาการดึงข้อมูล Forward/Backward และการทำงานของ Torch TB Profiler

เอกสารนี้สรุปแนวทางการศึกษาทำความเข้าใจการดึงข้อมูล forward และ backward pass ภายใน Torch TB Profiler Plugin รวมถึงการทำงานของไฟล์สำคัญที่เกี่ยวข้อง และแนวทางการศึกษาโค้ดโดยรวม

## 1. การดึงข้อมูล Forward และ Backward Pass

การที่ profiler สามารถแสดงผลข้อมูล forward และ backward pass แยกกันได้นั้น อาศัยการทำงานร่วมกันของไฟล์หลักๆ คือ `trace.py` และ `op_tree.py`

**กลไกหลัก:**

1.  **การสร้าง Map เชื่อมโยง Forward-Backward (`trace.py`)**:
    *   **ฟังก์ชัน `create_association_events(events)`**: เป็นหัวใจหลักในการสร้างความเชื่อมโยงเบื้องต้น
    *   **Input**: รับรายการ event ที่มี category (`cat`) เป็น `'fwdbwd'` ซึ่งเป็น event พิเศษที่ PyTorch profiler ใส่เข้ามาใน trace file เพื่อระบุคู่ของ forward และ backward operations
    *   **การทำงาน**:
        *   Event ที่มี phase (`ph`) เป็น `'s'` (start) จะถูกมองว่าเป็นจุดเริ่มต้นของ forward pass และจะถูกเก็บ timestamp (`ts`) และ correlation ID (`id`) ไว้ใน `forward_map`
        *   Event ที่มี phase (`ph`) เป็น `'f'` (finish/end) จะถูกมองว่าเป็นจุดสิ้นสุดของ backward pass ที่สอดคล้อง และจะถูกเก็บ timestamp และ correlation ID ไว้ใน `backward_map`
        *   จากนั้น ฟังก์ชันจะจับคู่ event จาก `forward_map` และ `backward_map` ที่มี correlation ID เดียวกัน
    *   **Output (`fwd_bwd_map`)**: คืนค่าเป็น dictionary ที่ key คือ timestamp ของ forward event และ value คือ timestamp ของ backward event ที่สัมพันธ์กัน Map นี้เป็นข้อมูลสำคัญที่ `op_tree.py` จะนำไปใช้

2.  **การสร้างโครงสร้าง Backward Pass ใน Op Tree (`op_tree.py` - คลาส `OpTreeBuilder`)**:
    *   `OpTreeBuilder` รับ `fwd_bwd_map` ที่สร้างจาก `trace.py` เข้ามา
    *   **ระบุ Module และ Backward Roots เดิม**: ค้นหา `ModuleNode` (แทน `nn.Module` ในโค้ด PyTorch) และ OperatorNode ที่เป็น backward pass เริ่มต้น (ชื่อมักจะขึ้นต้นด้วย `autograd::engine::evaluate_function:`)
    *   **ลบ Backward Roots เดิม (ถ้ามี Module)**: หากมีการใช้ `ModuleNode`, backward pass เดิมที่เป็นเพียง operator ทั่วไปจะถูกลบออกจาก op tree เพื่อเตรียมสร้าง `BackwardNode` ที่มีความหมายและโครงสร้างที่ดีกว่า
    *   **จัดกลุ่ม Backward Operations**: Backward operations ที่ต่อเนื่องกัน (เช่น `AccumulateGrad` ที่ตามหลัง backward op หลัก) จะถูกจัดกลุ่มเข้าด้วยกัน
    *   **เชื่อมโยง Forward กับกลุ่ม Backward Operations (`_get_backward_roots`)**: ใช้ `fwd_bwd_map` เพื่อระบุว่า "กลุ่มของ backward operations" ใด สัมพันธ์กับ "forward operation" (ที่ระบุด้วย timestamp) ใด ผลลัพธ์คือ `fwd_to_bwdroot` map (จาก forward timestamp ไปยัง *list ของ backward operation nodes*)
    *   **สร้าง `BackwardNode` (`_build_backward_module`)**:
        *   สำหรับแต่ละ `ModuleNode` ใน forward pass จะมีการสร้าง `BackwardNode` ที่สอดคล้องกัน (เช่น `MyModule.backward`)
        *   เมื่อวนลูปผ่าน children (ที่เป็น forward operator) ของ `ModuleNode`, จะใช้ timestamp ของ forward operator นั้นไปค้นหาใน `fwd_to_bwdroot` map เพื่อดึง *list ของ backward operation nodes* ที่เกี่ยวข้อง
        *   Backward operation nodes เหล่านี้จะถูกเพิ่มเข้าไปเป็น children ของ `BackwardNode` ที่กำลังสร้าง
    *   **แทรก `BackwardNode` เข้า Op Tree**: `BackwardNode` ที่สมบูรณ์ (มี children เป็น backward operations จริงๆ) จะถูกแทรกกลับเข้าไปใน op tree หลักในตำแหน่งที่เหมาะสม (ปกติจะอยู่หลัง forward pass ของ module ที่เกี่ยวข้อง)

**สรุปการดึงข้อมูล Fwd/Bwd**:
Profiler ไม่ได้ "ดึง" ข้อมูล forward/backward ออกมาตรงๆ จากจุดเดียว แต่เป็นกระบวนการ:
`trace.py` สร้าง "แผนที่" (`fwd_bwd_map`) จากข้อมูลที่ PyTorch profiler ให้มา -> `op_tree.py` ใช้ "แผนที่" นี้เพื่อ "สร้าง" และ "จัดระเบียบ" `BackwardNode` ที่มีความหมาย และนำ operation ที่เกี่ยวข้องจริงๆ มาใส่ในโครงสร้างนั้น ทำให้ op tree สุดท้ายสามารถแสดงผล forward และ backward pass ที่เชื่อมโยงกันได้

## 2. เจาะลึก `trace.py` และ `op_tree.py`

### `trace.py`

*   **หน้าที่หลัก**:
    *   นิยามคลาสสำหรับ Event ประเภทต่างๆ (เช่น `OperatorEvent`, `KernelEvent`) เพื่อจัดโครงสร้างข้อมูลจาก trace
    *   แปลงข้อมูล event ดิบ (จาก JSON) เป็น Event Object ที่ типизированный (typed)
    *   มี `EventTypeMap` สำหรับแมพ category จาก JSON ไปเป็น `EventTypes` ภายใน
    *   **สำคัญ**: มีฟังก์ชัน `create_association_events` ที่สร้าง `fwd_bwd_map` สำหรับเชื่อมโยง forward กับ backward pass โดยอาศัย event ที่มี `cat='fwdbwd'` และ correlation ID จาก PyTorch profiler

*   **เกี่ยวข้องกับ Fwd/Bwd อย่างไร**:
    *   `trace.py` เป็นผู้ **เตรียมข้อมูลพื้นฐาน** คือ `fwd_bwd_map` ซึ่งเป็น dictionary ที่ map timestamp ของ forward event กับ backward event ที่คู่กัน
    *   ไม่ได้สร้างโครงสร้าง node ของ forward/backward โดยตรง แต่ให้ข้อมูลตั้งต้นที่จำเป็นแก่ `op_tree.py`

### `op_tree.py` (คลาส `OpTreeBuilder`)

*   **หน้าที่หลัก**:
    *   สร้างโครงสร้าง Op Tree (ต้นไม้ของ operation) จาก `OperatorNode` ที่ได้จาก `event_parser.py`
    *   จัดการความสัมพันธ์พ่อ-ลูกของ operation ตามช่วงเวลา
    *   **สำคัญ**: ทำการ **Backward Correlation** คือสร้าง `BackwardNode` ที่มีความหมายสำหรับแต่ละ `ModuleNode` และนำ backward operation ที่แท้จริง (ที่ได้จากการใช้ `fwd_bwd_map`) มาเป็น children ของ `BackwardNode` นั้นๆ

*   **เกี่ยวข้องกับ Fwd/Bwd อย่างไร**:
    *   `op_tree.py` คือผู้ที่ **ใช้ `fwd_bwd_map` เพื่อสร้างโครงสร้าง forward และ backward pass ที่ชัดเจน** ใน Op Tree
    *   เมธอด `build_tree` และ helper functions เช่น `_get_backward_roots`, `_build_backward_module` เป็นส่วนสำคัญในการตีความ `fwd_bwd_map` และประกอบร่าง `BackwardNode` ที่เชื่อมโยงกับ forward pass ของแต่ละ module

## 3. แนวทางการศึกษาโค้ด (ทั่วไป)

เพื่อให้เข้าใจภาพรวมการทำงานของ profiler แนะนำให้ศึกษาตามลำดับนี้:

1.  **`loader.py` (`RunLoader`)**: จุดเริ่มต้นการโหลดข้อมูล trace
2.  **`data.py` (`RunProfileData`)**: ศูนย์กลางการเก็บข้อมูลและเรียกใช้ parser ต่างๆ
3.  **`trace.py`**: นิยามโครงสร้าง Event และการสร้าง `fwd_bwd_map`
4.  **`node.py`**: นิยามโครงสร้าง Node ต่างๆ ที่ใช้ใน Op Tree (`OperatorNode`, `DeviceNode` etc.)
5.  **`event_parser.py` (`EventParser`)**: การประมวลผล event เบื้องต้น, การแบ่ง step/role, การเรียก `OpTreeBuilder`
6.  **`op_tree.py` (`OpTreeBuilder`)**: การสร้าง Op Tree ที่สมบูรณ์ รวมถึง Backward Correlation
7.  **Parser เฉพาะทาง** (เช่น `overall_parser.py`, `op_agg.py`, `gpu_metrics_parser.py`): การวิเคราะห์ข้อมูลในแง่มุมต่างๆ
8.  **`communication.py`**: การจัดการ communication event (สำหรับ distributed)
9.  **`run_generator.py` (`RunGenerator`)**: การเตรียมข้อมูลสุดท้ายสำหรับแสดงผลใน TensorBoard UI

## 4. แนวทางการศึกษาโค้ด (เฉพาะ Forward/Backward)

หากต้องการเน้นเฉพาะการทำความเข้าใจส่วน forward/backward pass:

1.  **`trace.py` (ฟังก์ชัน `create_association_events`)**: ทำความเข้าใจการสร้าง `fwd_bwd_map` จาก event `cat='fwdbwd'`
2.  **`data.py` (ส่วนที่เรียก `create_association_events`)**: ดูว่า `fwd_bwd_map` ถูกส่งต่ออย่างไร
3.  **`event_parser.py` (ส่วนที่เรียก `OpTreeBuilder.build_tree`)**: ดูว่า `fwd_bwd_map` ถูกส่งไปให้ `OpTreeBuilder` อย่างไร
4.  **`op_tree.py` (`OpTreeBuilder` - เมธอด `build_tree` และ helpers ที่เกี่ยวข้อง)**:
    *   **`_get_modules()`**: การระบุ `ModuleNode` และ backward roots เดิม
    *   **`_get_backward_roots(fwd_bwd_map, ...)`**: **สำคัญมาก** - การใช้ `fwd_bwd_map` เพื่อ map forward op timestamp ไปยังกลุ่มของ backward op nodes
    *   **`_build_backward_module(...)`**: **สำคัญมาก** - การสร้าง `BackwardNode` และการนำ backward op nodes จริงๆ มาเป็น children โดยใช้ map จากขั้นตอนก่อน
    *   **`_insert_backward_modules()`**: การนำ `BackwardNode` ที่สร้างเสร็จแล้วไปใส่ใน Op Tree หลัก
5.  **`node.py` (คลาส `BackwardNode`, `ModuleNode`)**: ทำความเข้าใจโครงสร้างของ Node ที่เกี่ยวข้อง

หวังว่าข้อมูลนี้จะเป็นประโยชน์ต่อการศึกษาทำความเข้าใจโค้ด Torch TB Profiler ครับ
