# คำอธิบายโค้ด `profiler/op_tree.py` เชิงลึก

เอกสารนี้อธิบายการทำงานของไฟล์ `op_tree.py` จาก PyTorch Profiler Plugin อย่างละเอียด เพื่อให้เกิดความเข้าใจในวัตถุประสงค์, การไหลของข้อมูล, และตรรกะการทำงานของแต่ละส่วน สำหรับนำไปศึกษาและปรับใช้

## ภาพรวม

ไฟล์ `op_tree.py` มีคลาสเดียวคือ `OpTreeBuilder` ซึ่งมีหน้าที่ **สำคัญและซับซ้อนมาก** คือการแปลง flat list ของ `OperatorNode` (ที่ถูกสร้างและจัดกลุ่มตาม Thread ID โดย `EventParser`) ให้กลายเป็น **โครงสร้างต้นไม้ (tree) ที่มีความสัมพันธ์แบบ parent-child**. โครงสร้างนี้สะท้อน call stack การทำงานของโปรแกรม ทำให้สามารถวิเคราะห์ได้ว่า operator ใดเรียก operator ใด และใช้เวลาไปเท่าไหร่.

นอกจากนี้ `OpTreeBuilder` ยังมีตรรกะพิเศษในการ **"สังเคราะห์" (synthesize) `BackwardNode`** ขึ้นมาใหม่. เนื่องจาก profiler trace ไม่ได้ให้ข้อมูล `BackwardNode` มาโดยตรง, `OpTreeBuilder` จะใช้ข้อมูลความสัมพันธ์ forward-backward (`fwd_bwd_map`) เพื่อจัดกลุ่ม backward operations ที่กระจัดกระจายอยู่ ให้มาอยู่ภายใต้ `BackwardNode` ที่สอดคล้องกับ `ModuleNode` ใน forward pass.

---

## `class OpTreeBuilder`

### `__init__(self)`

*   **วัตถุประสงค์**: Constructor ของคลาส, ประกาศ instance variables.
*   **คำอธิบายโค้ด**:
    ```python
    # self.main_tid: จะใช้เก็บ Thread ID ของ main thread (thread ที่รัน `ProfilerStep#...`)
    # เพื่อใช้เป็นจุดอ้างอิงหลักในการแทรก node บางประเภท
    self.main_tid: int = None
    # self.tid2tree: Dict[int, OperatorNode] - จะใช้เก็บผลลัพธ์สุดท้าย
    # คือ operator tree ของแต่ละ thread, key คือ thread ID, value คือ root node ของ tree นั้น
    self.tid2tree: Dict[int, OperatorNode] = None
    ```

### `build_tree(self, tid2list, tid2zero_rt_list, staled_device_nodes, fwd_bwd_map)`

*   **วัตถุประสงค์**: เป็น public method และเป็น entry point หลักที่ `EventParser` เรียกใช้. ทำหน้าที่ orchestrate การสร้าง tree ทั้งหมด, รวมถึงการสังเคราะห์ backward nodes.
*   **Input Arguments**:
    *   `tid2list`: `Dict[int, List[OperatorNode]]` - flat list ของ `OperatorNode` ทั้งหมด แยกตาม Thread ID.
    *   `tid2zero_rt_list`: `List[RuntimeNode]` - runtime events ที่ไม่มี op แม่ (external_id=0).
    *   `staled_device_nodes`: `List[DeviceNode]` - device events (kernel/memcpy) ที่ไม่มี runtime แม่.
    *   `fwd_bwd_map`: `Dict[int, int]` - map ความสัมพันธ์ forward-backward ที่สร้างโดย `trace.create_association_events`.
*   **การทำงานและ Data Flow**:

    1.  **`self.tid2tree = self._build_tree(tid2list, ...)`**:
        *   **Action**: เรียก `_build_tree` (private method) เพื่อสร้าง operator tree พื้นฐานสำหรับทุก thread.
        *   **Output**: `self.tid2tree` จะมี tree พื้นฐานที่ยังไม่มีการจัดการ `BackwardNode` แบบพิเศษ.

    2.  **`if not fwd_bwd_map: ... return self.tid2tree`**:
        *   **Action**: ตรวจสอบ `fwd_bwd_map`. ถ้าไม่มีข้อมูลความสัมพันธ์ forward-backward, ก็ไม่จำเป็นต้องทำตรรกะที่ซับซ้อนต่อไป.
        *   **Output**: คืนค่า tree พื้นฐานกลับไปทันที.

    3.  **`self._set_main_tid()`**:
        *   **Action**: หา TID ของ main thread. ถ้าหาไม่เจอ (เช่น ไม่มี `ProfilerStep`), จะเดาโดยเลือก thread ที่มีระยะเวลาการทำงานยาวที่สุด.

    4.  **`modules, backward_nodes = self._get_modules()`**:
        *   **Action**: เรียก `_get_modules` เพื่อ:
            *   ค้นหา `ModuleNode` ทั้งหมดใน tree.
            *   ค้นหา `OperatorNode` ที่เป็นจุดเริ่มต้นของ backward pass (ชื่อขึ้นต้นด้วย `autograd::engine::evaluate_function:`).
            *   **ถ้าเจอ `ModuleNode`**, จะทำการ **"ถอด" backward node เดิมออกจาก tree** (`p.children = [c for c in p.children if c not in nodes]`) เพื่อเตรียมที่จะสร้าง `BackwardNode` ใหม่เข้าไปแทนที่.
        *   **Output**: `modules` (list ของ `ModuleNode`) และ `backward_nodes` (flat list ของ backward op ที่ถูกถอดออกมา).

    5.  **`_, ts2parent = OpTreeBuilder._get_node_parents(...)`**:
        *   **Action**: สร้าง map `timestamp -> parent_node` สำหรับ backward nodes ที่ได้มา.

    6.  **`agg_nodes = OpTreeBuilder._group_backward_nodes(...)`**:
        *   **Action**: จัดกลุ่ม backward nodes. โดยปกติ `AccumulateGrad` op จะตามหลัง backward op หลัก, เมธอดนี้จะรวมมันเข้าไว้ด้วยกัน.

    7.  **`fwd_bwd_root = self._get_backward_roots(...)`**:
        *   **Action**: **นี่คือส่วนที่ใช้ `fwd_bwd_map` อย่างจริงจัง.** วนลูป `fwd_bwd_map` เพื่อเชื่อมโยง forward op (`ModuleNode`'s children) กับ backward op ที่คู่กัน.
        *   **Output**: สร้าง `fwd_bwd_root` ซึ่งเป็น dictionary ที่ map timestamp ของ forward op ไปยัง list ของ backward op nodes ที่จัดกลุ่มแล้ว.

    8.  **`for module in modules: OpTreeBuilder._build_backward_module(...)`**:
        *   **Action**: เรียก `_build_backward_module` แบบ recursive เพื่อสร้าง `BackwardNode` tree ขึ้นมาใหม่โดยอิงจากโครงสร้างของ `ModuleNode` tree. มันจะนำ backward op ที่หาได้จากขั้นตอนที่ 7 มาใส่เป็น children ของ `BackwardNode` ที่สร้างขึ้น.
        *   **Output**: `backward_modules` ซึ่งเป็น list ของ `BackwardNode` ระดับบนสุด.

    9.  **`OpTreeBuilder._insert_backward_modules(...)`**:
        *   **Action**: นำ `BackwardNode` tree ที่สร้างเสร็จแล้ว "แทรก" กลับเข้าไปใน operator tree หลักในตำแหน่งที่เหมาะสม.

    10. **Return `self.tid2tree`**: คืนค่า `tid2tree` ที่มีโครงสร้างสมบูรณ์ (รวม backward nodes ที่สังเคราะห์ขึ้นใหม่).

### `_build_tree(self, ...)`

*   **วัตถุประสงค์**: เมธอดภายในที่วนลูปทุก thread ID เพื่อเรียก `_build_tree_internal` และสร้าง tree สำหรับแต่ละ thread.

### `_build_tree_internal(self, host_node_list, ...)`

*   **วัตถุประสงค์**: สร้างโครงสร้าง tree พื้นฐานจาก flat list ของ `OperatorNode` สำหรับ thread เดียว.
*   **การทำงาน**:
    *   **`build_tree_relationship(...)`**:
        *   **ใช้ Stack-based Algorithm (สำคัญมาก):**
            1.  สร้าง `root_node` เสมือนขึ้นมาและ push เข้า stack.
            2.  วนลูปผ่าน `host_node_list` (ที่เรียงตามเวลาเริ่มต้นแล้ว).
            3.  สำหรับแต่ละ `node`, จะเปรียบเทียบ `start_time` และ `end_time` กับ node ที่อยู่บนสุดของ stack (`tail_node`).
            4.  `while True:`:
                *   `if node.start_time < tail_node.end_time:`: ถ้าเวลาเริ่มต้นของ `node` อยู่ก่อนเวลาสิ้นสุดของ `tail_node`, แสดงว่า `node` อาจเป็น child.
                    *   `if node.end_time <= tail_node.end_time:`: **เงื่อนไขการเป็น Child ที่ถูกต้อง.** `node` ทั้งหมดต้องอยู่ภายในขอบเขตเวลาของ `tail_node`.
                    *   `tail_node.children.append(node)`: เพิ่มเป็น child.
                    *   `node_stack.append(node)`: push `node` ปัจจุบันเข้า stack เพื่อเป็น parent ของ node ถัดไป.
                    *   `break`: ออกจาก `while` loop เพราะหาที่ของ `node` ได้แล้ว.
                *   `else:`: ถ้าเวลาเริ่มต้นของ `node` อยู่หลังหรือเท่ากับเวลาสิ้นสุดของ `tail_node`, แสดงว่า `tail_node` ทำงานจบแล้ว และ `node` ไม่ใช่ child.
                    *   `node_stack.pop()`: pop `tail_node` ออกจาก stack แล้ววน `while` loop อีกครั้งเพื่อเปรียบเทียบกับ parent ใหม่ที่อยู่บนสุดของ stack.
    *   **`remove_dup_nodes(node: OperatorNode)`**:
        *   หลังจากสร้าง tree เสร็จ, จะมีการเรียกฟังก์ชันนี้แบบ recursive เพื่อรวม operator ที่ชื่อเหมือนกันและเรียกต่อกันทันทีให้เป็น node เดียว (เช่น `conv -> conv` จะยุบเหลือ `conv` เดียว). นี่เป็นการทำความสะอาด tree ให้เหมือนกับผลลัพธ์ของ autograd profiler.
    *   **`root_node.fill_stats()`**:
        *   เรียกเมธอดของ `OperatorNode` เพื่อคำนวณค่าสถิติต่างๆ (เช่น `self_cpu_time`, `device_duration`) สำหรับทุก node ใน tree.

### เมธอดสำหรับจัดการ Backward Pass

กลุ่มของเมธอด (`_get_modules`, `_get_node_parents`, `_group_backward_nodes`, `_get_backward_roots`, `_build_backward_module`, `_insert_backward_modules`) ทำงานร่วมกันเพื่อทำกระบวนการที่ซับซ้อนในการ "สังเคราะห์" `BackwardNode` และแทรกกลับเข้าไปใน tree หลัก.

*   **`_get_modules`**: หา `ModuleNode` และ backward op ดั้งเดิม และ "ถอด" backward op ออกจาก tree.
*   **`_group_backward_nodes`**: จัดกลุ่ม backward op กับ `AccumulateGrad` ที่ตามมา.
*   **`_get_backward_roots`**: ใช้ `fwd_bwd_map` เพื่อหาว่า backward op กลุ่มไหนเป็นของ forward op ตัวไหน.
*   **`_build_backward_module`**: สร้าง `BackwardNode` tree ขึ้นมาใหม่ให้มีโครงสร้างเหมือน `ModuleNode` tree.
*   **`_insert_backward_modules`**: แทรก `BackwardNode` tree ที่สร้างเสร็จแล้วกลับเข้าไปใน tree หลัก.

**ผลลัพธ์สุดท้าย**: `OpTreeBuilder` จะคืนค่า dictionary (`tid2tree`) ที่ value เป็น root `OperatorNode` ของแต่ละ thread. tree นี้มีโครงสร้าง parent-child ที่สมบูรณ์, มีการคำนวณสถิติเบื้องต้น, และมีการสังเคราะห์ `BackwardNode` เพื่อให้ง่ายต่อการวิเคราะห์และแสดงผล.
