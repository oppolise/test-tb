# อธิบายโค้ด `op_tree.py` เชิงลึก

เอกสารนี้จะอธิบายการทำงานของไฟล์ `tb_plugin/torch_tb_profiler/profiler/op_tree.py` อย่างละเอียด เพื่อให้เข้าใจถึงวิธีการที่ไฟล์นี้ทำหน้าที่เป็น **"สถาปนิก" ที่สร้างโครงสร้าง Op Tree (ต้นไม้ของ Operation) ที่สมบูรณ์และมีความหมาย** จากรายการ event ที่ยังไม่มีความสัมพันธ์กัน

## ภาพรวมของ `op_tree.py`

`op_tree.py` มีคลาสหลักคือ `OpTreeBuilder` ซึ่งเป็นหัวใจสำคัญในการแปลงรายการ `OperatorNode` ที่เรียงตามเวลา (จาก `event_parser.py`) ให้กลายเป็นโครงสร้างต้นไม้ที่มีลำดับชั้น (hierarchy) แสดงให้เห็นว่า operation ใดเรียกใช้ operation ใด (parent-child relationship)

หน้าที่หลักและซับซ้อนที่สุดของ `OpTreeBuilder` คือ **การสร้างและเชื่อมโยง Backward Pass (Backward Correlation)** โดยใช้ `fwd_bwd_map` ที่ได้จาก `trace.py` เพื่อสร้าง `BackwardNode` ที่มีความหมาย และนำไปแทรกใน Op Tree ให้ถูกต้อง

---

## การทำงานของคลาส `OpTreeBuilder`

### `__init__(self)`

*   **หน้าที่**: Constructor ของคลาส, ประกาศตัวแปร instance
*   **โค้ด**:
    ```python
    class OpTreeBuilder:
        BACKWARD_ROOT_PREFIX = 'autograd::engine::evaluate_function:'
        BACKWARD_ACCUMULATE_GRAD = 'autograd::engine::evaluate_function: torch::autograd::AccumulateGrad'

        def __init__(self):
            self.main_tid: int = None
            self.tid2tree: Dict[int, OperatorNode] = None
    ```
*   **คำอธิบาย**:
    *   `BACKWARD_ROOT_PREFIX`, `BACKWARD_ACCUMULATE_GRAD`: กำหนดค่าคงที่ที่เป็น string สำหรับใช้ระบุชื่อของ backward operation ที่สำคัญ
    *   `self.main_tid`: จะเก็บ Thread ID ของ thread หลัก (main thread)
    *   `self.tid2tree`: **สำคัญมาก** - เป็น dictionary ที่จะเก็บ Op Tree ที่สร้างเสร็จแล้วของแต่ละ Thread ID

### `build_tree(self, tid2list, tid2zero_rt_list, staled_device_nodes, fwd_bwd_map)`

*   **หน้าที่**: เป็นเมธอดหลักและเป็น public interface ของคลาสนี้ ควบคุมกระบวนการสร้าง Op Tree ทั้งหมด
*   **โค้ดและคำอธิบายเชิงลึก**:

    **ส่วนที่ 1: สร้าง Op Tree พื้นฐาน**
    ```python
    def build_tree(self, ..., fwd_bwd_map: Dict[int, int]):
        """Construct the BackwardNode and replace the original backward nodes"""
        self.tid2tree = self._build_tree(tid2list, tid2zero_rt_list, staled_device_nodes)
    ```
    *   **การทำงาน**: เรียก `_build_tree` ซึ่งเป็น private method เพื่อสร้าง Op Tree พื้นฐานสำหรับแต่ละ thread ก่อน โดยยังไม่มีการจัดการ backward pass ที่ซับซ้อน
    *   **ผลลัพธ์**: `self.tid2tree` จะมี Op Tree ที่มีโครงสร้างพ่อ-ลูกตามช่วงเวลา แต่ backward pass ยังเป็นเพียง `OperatorNode` ธรรมดา

    **ส่วนที่ 2: ตรวจสอบและเริ่มกระบวนการ Backward Correlation**
    ```python
    if not fwd_bwd_map:
        logger.debug('there is no any forward backward association, skip processing backward correlation.')
        return self.tid2tree
    ```
    *   **การทำงาน**: ตรวจสอบว่ามี `fwd_bwd_map` (ที่ได้จาก `trace.py`) หรือไม่ ถ้าไม่มี (หมายถึงไม่มีข้อมูลเชื่อมโยง forward/backward) ก็จะคืนค่า Op Tree พื้นฐานไปเลยและจบการทำงาน

    **ส่วนที่ 3: เตรียมการสำหรับ Backward Correlation**
    ```python
    self._set_main_tid()

    modules, backward_nodes = self._get_modules()
    if not modules or not backward_nodes:
        return self.tid2tree
    ```
    *   `self._set_main_tid()`: หา Thread ID ของ thread หลัก (มักจะเป็น thread ที่มี `ProfilerStep#` event)
    *   `modules, backward_nodes = self._get_modules()`: **ขั้นตอนสำคัญ** - เรียกเมธอด `_get_modules` เพื่อ:
        1.  ค้นหา `ModuleNode` ทั้งหมดใน Op Tree
        2.  ค้นหา `OperatorNode` ที่เป็น backward pass เริ่มต้น (ชื่อขึ้นต้นด้วย `BACKWARD_ROOT_PREFIX`)
        3.  **ถ้าเจอ `ModuleNode`**, จะทำการลบ backward node เดิมออกจาก Op Tree เพื่อเตรียมสร้าง `BackwardNode` ใหม่เข้าไปแทนที่
    *   ถ้าไม่เจอ `ModuleNode` หรือ `backward_nodes` ก็จะจบการทำงาน

    **ส่วนที่ 4: สร้างและแทรก Backward Modules**
    ```python
    _, ts2parent = OpTreeBuilder._get_node_parents(backward_nodes)
    agg_nodes = OpTreeBuilder._group_backward_nodes(backward_nodes)
    fwd_bwd_root = self._get_backward_roots(fwd_bwd_map, ts2parent, agg_nodes)
    # ...

    backward_modules: List[BackwardNode] = []
    for module in modules:
        OpTreeBuilder._build_backward_module(module, None, fwd_bwd_root, backward_modules)
    OpTreeBuilder._insert_backward_modules(self.tid2tree[self.main_tid], backward_modules)
    ```
    *   **การทำงาน**: นี่คือกระบวนการหลักของการทำ Backward Correlation:
        1.  `_get_node_parents`, `_group_backward_nodes`: เป็นเมธอด helper เพื่อจัดระเบียบและจัดกลุ่ม backward node ที่หาเจอ
        2.  `fwd_bwd_root = self._get_backward_roots(...)`: **ใช้ `fwd_bwd_map`** เพื่อสร้าง dictionary (`fwd_bwd_root`) ที่ map timestamp ของ forward op ไปยัง *กลุ่มของ backward operation nodes* ที่สอดคล้องกัน
        3.  `for module in modules: ...`: วนลูปผ่าน `ModuleNode` แต่ละตัวที่หาเจอ
        4.  `_build_backward_module(...)`: เรียกเมธอดนี้เพื่อ **สร้าง `BackwardNode`** ที่สมบูรณ์สำหรับแต่ละ `module` โดยใช้ `fwd_bwd_root` map เพื่อดึง backward op ที่ถูกต้องมาใส่เป็น children
        5.  `_insert_backward_modules(...)`: นำ `BackwardNode` ทั้งหมดที่สร้างเสร็จแล้ว แทรกกลับเข้าไปใน Op Tree ของ thread หลัก

    **ส่วนที่ 5: คืนค่า Op Tree ที่สมบูรณ์**
    ```python
    self.tid2tree = {tid: root for tid, root in self.tid2tree.items() if len(root.children) > 0}
    return self.tid2tree
    ```
    *   **การทำงาน**: กรองเอาเฉพาะ tree ที่มี children (ไม่เอา tree ว่าง) และคืนค่า `self.tid2tree` ที่ผ่านการทำ Backward Correlation เรียบร้อยแล้ว

### `_build_tree(self, tid2list, ...)`

*   **หน้าที่**: สร้าง Op Tree พื้นฐานสำหรับแต่ละ thread ID
*   **การทำงาน**:
    1.  วนลูปผ่าน `tid2list` (dictionary ที่มี key เป็น `tid` และ value เป็น list ของ `OperatorNode`)
    2.  `op_list.sort(key=lambda x: (x.start_time, -x.end_time))`: **สำคัญมาก** - จัดเรียง `OperatorNode` ตามเวลาเริ่มต้น (น้อยไปมาก) และถ้าเวลาเริ่มต้นเท่ากัน จะให้ตัวที่เวลาสิ้นสุดมากกว่า (กินเวลานานกว่า) มาก่อน ซึ่งจำเป็นสำหรับการสร้างโครงสร้างพ่อ-ลูกที่ถูกต้อง
    3.  เรียก `_build_tree_internal` เพื่อสร้าง tree สำหรับ `tid` นั้นๆ
    4.  เก็บ tree ที่ได้ไว้ใน `tid2tree`

### `_build_tree_internal(self, host_node_list, ...)`

*   **หน้าที่**: สร้างโครงสร้างต้นไม้ (parent-child relationship) จากรายการ node ที่เรียงตามเวลาแล้ว
*   **โค้ดและคำอธิบายเชิงลึก**:
    *   **`build_tree_relationship(...)` (ฟังก์ชันภายใน)**:
        ```python
        def build_tree_relationship(...):
            # ...
            node_stack: List[OperatorNode] = []
            root_node = OperatorNode(name='CallTreeRoot', ...)
            node_stack.append(root_node)
            for node in host_node_list:
                while True:
                    tail_node = node_stack[-1]
                    if node.start_time < tail_node.end_time:
                        if node.end_time <= tail_node.end_time:
                            tail_node.children.append(node)
                            node_stack.append(node)
                        # ... (error logging) ...
                        break
                    else:
                        node_stack.pop()
            return root_node
        ```
        1.  **ใช้ Stack**: สร้าง `node_stack` เพื่อติดตาม "parent" ปัจจุบัน
        2.  **สร้าง Root Node**: สร้าง `root_node` เสมือนชื่อ `CallTreeRoot` ที่มีช่วงเวลากว้างที่สุด และใส่ลงใน stack
        3.  **วนลูปผ่าน Node**:
            *   สำหรับแต่ละ `node` ที่เข้ามา, จะดูที่ `tail_node` (node บนสุดของ stack ซึ่งคือ parent ปัจจุบัน)
            *   **ถ้า `node` อยู่ภายใน `tail_node`**: (`node.start_time < tail_node.end_time` และ `node.end_time <= tail_node.end_time`) หมายความว่า `node` เป็น child ของ `tail_node`
                *   เพิ่ม `node` เข้าไปใน `tail_node.children`
                *   push `node` เข้าไปใน stack (เพราะมันจะกลายเป็น parent สำหรับ node ต่อไปที่อาจจะอยู่ภายในมัน)
                *   `break` ออกจาก `while True` เพื่อไปทำ `node` ตัวถัดไป
            *   **ถ้า `node` ไม่ได้อยู่ภายใน `tail_node`**: (`node.start_time >= tail_node.end_time`) หมายความว่า `tail_node` ทำงานจบแล้ว
                *   `node_stack.pop()`: เอา `tail_node` ออกจาก stack เพื่อกลับไปยัง parent ในระดับที่สูงขึ้น
                *   วน `while` loop ต่อไปเพื่อเปรียบเทียบ `node` กับ parent ตัวใหม่
    *   **`remove_dup_nodes(...)`**: เรียกฟังก์ชัน helper เพื่อรวม node ที่มีชื่อเดียวกันและเรียกต่อกันทันที (เช่น `conv2d` -> `conv2d`) ให้เป็น node เดียว
    *   **`root_node.fill_stats()`**: เรียก `fill_stats()` (จาก `node.py`) เพื่อคำนวณค่า duration ต่างๆ แบบ recursive
    *   **คืนค่า `root_node`**: คืนค่า Op Tree ที่สร้างเสร็จแล้ว

### เมธอดสำหรับ Backward Correlation (อธิบายการทำงานร่วมกัน)

*   **`_get_modules()`**: ทำหน้าที่ "สำรวจและเตรียมพื้นที่" โดยการหา `ModuleNode` และ `backward_nodes` เริ่มต้น และที่สำคัญคือ **ลบ `backward_nodes` เดิมออกจาก tree** เพื่อให้มีที่สำหรับ `BackwardNode` ใหม่ที่จะสร้างขึ้น
*   **`_get_backward_roots(fwd_bwd_map, ...)`**: ทำหน้าที่ "สร้างแผนที่เชื่อมโยง" โดยใช้ `fwd_bwd_map` จาก `trace.py` เพื่อแปลงจาก `(forward_ts -> backward_ts)` ให้กลายเป็น `(forward_ts -> list_of_backward_op_nodes)` ซึ่งเป็นข้อมูลที่พร้อมใช้งานมากขึ้น
*   **`_build_backward_module(module, ..., fwd_bwd_map, ...)`**: ทำหน้าที่ "สร้างและประกอบร่าง" โดย:
    1.  สร้าง `BackwardNode` เปล่าๆ สำหรับ `module`
    2.  วนลูปผ่าน children ของ `module` (ซึ่งเป็น forward op)
    3.  ใช้ timestamp ของ forward op ไปค้นหาใน "แผนที่เชื่อมโยง" (`fwd_bwd_map` ที่ตอนนี้คือ `fwd_to_bwdroot`) เพื่อดึง *list ของ backward op nodes* ที่ถูกต้อง
    4.  นำ backward op nodes ที่ได้มาใส่เป็น children ของ `BackwardNode` ที่สร้างไว้
*   **`_insert_backward_modules(root, ...)`**: ทำหน้าที่ "ติดตั้ง" โดยนำ `BackwardNode` ที่ประกอบร่างเสร็จแล้ว ไปแทรกใน Op Tree หลักในตำแหน่งที่ถูกต้องตามลำดับเวลา

---
หวังว่าคำอธิบายนี้จะช่วยให้เข้าใจบทบาทและการทำงานของ `op_tree.py` ในฐานะสถาปนิกผู้สร้าง Op Tree ที่สมบูรณ์และมีความหมายจากข้อมูลดิบได้ครับ
