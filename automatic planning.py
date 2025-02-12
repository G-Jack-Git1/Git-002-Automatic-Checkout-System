import cv2
import numpy as np
import heapq


class MultiPointPathPlanner:
    def __init__(self, image_path):
        # 初始化地图和参数
        self.original = cv2.imread(image_path)
        self.gray = cv2.cvtColor(self.original, cv2.COLOR_BGR2GRAY)
        _, self.binary = cv2.threshold(self.gray, 250, 255, cv2.THRESH_BINARY)
        #将灰度图像转换为二值图像，阈值为250

        self.points = []  # 存储所有点坐标
        self.all_paths = []  # 存储所有路径段
        self.total_distance = 0  # 总路径长度
        self.drawing = self.original.copy()

        # 设置窗口和鼠标回调
        cv2.namedWindow('Multi-Point Path Planner')
        cv2.setMouseCallback('Multi-Point Path Planner', self.mouse_handler)

    class Node:
        # 路径规划中的节点
        def __init__(self, pos, parent=None):
            # 路径规划中的节点
            self.pos = pos  # 节点位置(x, y)
            self.parent = parent # 指向父节点的引用，用于回溯路径。
            self.g = 0  # 实际代价
            self.h = 0  # 启发代价
            self.f = 0  # 总代价

        def __lt__(self, other):
            #比较两个节点的总代价
            return self.f < other.f

    def heuristic(self, a, b):
        # 改进的启发函数（对角线距离）
        dx = abs(a[0] - b[0])
        dy = abs(a[1] - b[1])
        return (dx + dy) + (np.sqrt(2) - 2) * min(dx, dy) # 计算从节点a到节点b的的曼哈顿修正值作为启发代价

    def astar(self, start, end):
        # A* 路径规划算法实现
        open_heap = []  # 存储待探索的节点及其总代价的最小堆
        closed_dict = {}  # 存储已探索的节点
        directions = [
            (0, 1, 1), (1, 0, 1), (0, -1, 1), (-1, 0, 1),
            (1, 1, np.sqrt(2)), (1, -1, np.sqrt(2)),
            (-1, 1, np.sqrt(2)), (-1, -1, np.sqrt(2))
        ]  # 8个方向的移动向量和对应的代价

        # 初始化起始节点和目标节点，并将起始节点添加到优先队列open_heap中
        start_node = self.Node(start)
        end_node = self.Node(end)
        heapq.heappush(open_heap, (start_node.f, start_node))

        while open_heap:
            current = heapq.heappop(open_heap)[1] # 从优先队列open_heap中取出代价最小的节点

            # 如果已到达目标节点，则返回路径
            if current.pos == end_node.pos:
                path = []

                # 生成路径
                while current:
                    path.append(current.pos)
                    current = current.parent
                return path[::-1]

            # 选择代价最小的节点
            if current.pos in closed_dict:
                if closed_dict[current.pos].f <= current.f:
                    continue
            closed_dict[current.pos] = current

            for dx, dy, cost in directions: # 遍历8个方向的移动向量和对应的代价，生成邻居节点

                # 计算相邻节点的位置
                x = current.pos[0] + dx
                y = current.pos[1] + dy

                if not (0 <= x < self.binary.shape[1] and 0 <= y < self.binary.shape[0]):
                    # 检查邻居节点的坐标(x, y)是否在地图范围内
                    continue
                if self.binary[y, x] == 255:
                    # 检查邻居节点是否为障碍物
                    continue

                # 生成当前节点的邻居节点，并计算邻居节点的代价
                neighbor = self.Node((x, y), current)
                neighbor.g = current.g + cost
                neighbor.h = self.heuristic(neighbor.pos, end_node.pos)
                neighbor.f = neighbor.g + neighbor.h

                #
                add_to_heap = True
                # 检查邻居节点是否已经在open_heap中，并且其总代价f是否小于等于当前邻居节点的总代价
                for _, node in open_heap:
                    if node.pos == neighbor.pos and node.f <= neighbor.f:
                        add_to_heap = False
                        break
                # 判断是否将邻居节点添加到open_heap中
                if add_to_heap:
                    heapq.heappush(open_heap, (neighbor.f, neighbor))

        return None

    def mouse_handler(self, event, x, y, flags, param):
        # 定义鼠标左键按下事件处理内容（添加路径点）
        if event == cv2.EVENT_LBUTTONDOWN:
            # 障碍物检测
            if self.binary[y, x] == 255:
                print("错误：不能选择障碍物区域！")
                return

            # 添加新点
            self.points.append((x, y))
            print(f"已添加第 {len(self.points)} 个点: ({x}, {y})")

            # 绘制当前点，绿色圆点
            cv2.circle(self.drawing, (x, y), 8, (0, 255, 0), -1)

            # 当有至少两个点时计算路径
            if len(self.points) >= 2:
                start = self.points[-2]
                end = self.points[-1]
                path = self.astar(start, end)

                if path:
                    # 绘制路径，蓝色
                    for i in range(1, len(path)):
                        cv2.line(self.drawing, path[i - 1], path[i], (255, 0, 0), 2)

                    # 计算路径长度
                    segment_length = sum(np.linalg.norm(np.array(path[i]) - np.array(path[i - 1]))
                                         for i in range(1, len(path)))
                    self.total_distance += segment_length
                    self.all_paths.extend(path)
                else:
                    print("警告：未找到有效路径！")
                    self.points.pop()  # 移除无效点

            # 更新显示
            cv2.putText(self.drawing, f"总长度: {self.total_distance:.2f}px",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.imshow('Multi-Point Path Planner', self.drawing)

        # 定义鼠标右键按下事件处理内容（撤销路径点）
        elif event == cv2.EVENT_RBUTTONDOWN:  # 右键撤销
            if self.points:
                self.points.pop()
                print(f"已移除第 {len(self.points)} 个点")
                self.redraw_all() # 重新绘制所有点和路径

    def redraw_all(self):
        # 重新绘制所有点和路径
        self.drawing = self.original.copy()
        self.total_distance = 0
        self.all_paths = []

        # 重新计算所有路径
        for i in range(len(self.points)):
            # 绘制点，起点红色，终点蓝色，中间路径点绿色
            cv2.circle(self.drawing, self.points[i], 8,
                       (0, 255, 0) if i == 0 else (0, 0, 255) if i == len(self.points) - 1 else (255, 0, 0),
                       -1)

            if i > 0:
                path = self.astar(self.points[i - 1], self.points[i])
                if path:
                    for j in range(1, len(path)):
                        cv2.line(self.drawing, path[j - 1], path[j], (255, 0, 0), 2)
                    self.total_distance += sum(np.linalg.norm(np.array(path[k]) - np.array(path[k - 1]))
                                               for k in range(1, len(path)))

        # 更新显示
        cv2.putText(self.drawing, f"总长度: {self.total_distance:.2f}px",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.imshow('Multi-Point Path Planner', self.drawing)

    def run(self):
        print("操作指南：")
        print("1. 左键点击添加路径点")
        print("2. 右键点击撤销上一个点")
        print("3. 按ESC键退出程序")

        while True:
            cv2.imshow('Multi-Point Path Planner', self.drawing)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC键退出
                break
        cv2.destroyAllWindows()


if __name__ == "__main__":
    planner = MultiPointPathPlanner("photo1.png")  # 替换为你的图像路径
    planner.run()