#include <bits/stdc++.h>
#include <Eigen/Eigen>

struct Node {
    int x, y;        // 节点所在的网格坐标
    double g_cost;   // 从起点到当前节点的代价
    double h_cost;   // 从当前节点到终点的估计代价
    std::shared_ptr<Node> parent;    // 父节点，用于回溯路径

    Node(int x, int y, double g_cost, double h_cost, std::shared_ptr<Node> parent = nullptr)
            : x(x), y(y), g_cost(g_cost), h_cost(h_cost), parent(std::move(parent)) {}
    Node(){};
    double f() const { return g_cost + h_cost; } // 总代价值

};
// 比较器，用于优先队列
struct cmp{
    bool operator()(std::shared_ptr<Node> a, std::shared_ptr<Node> b){
        return a->f() > b->f();
    }

};
struct GridMap {
    int width;
    int height;
    double map_max;
    double map_min;
    double grid_resolution;
    std::vector<std::vector<int>> grid; // 0: 空闲, 1: 占用

    GridMap(int w, int h, double map_min_, double map_max_, double res) : width(w), height(h), map_min(map_min_), map_max(map_max_), grid_resolution(res), grid(w, std::vector<int>(h, 0)) {}
    GridMap(){};
    void markObstacle(double cx, double cy, double radius) {
        int grid_cx = std::round((cx - map_min) / grid_resolution);
        int grid_cy = std::round((cy - map_min) / grid_resolution);
        int grid_radius = std::round(radius / grid_resolution);
        // Step 1: 将圆形区域标记为占用
            // your code
            for (int dx = -grid_radius; dx <= grid_radius; ++dx) {
                for (int dy = -grid_radius; dy <= grid_radius; ++dy) {
                    int nx = grid_cx + dx;
                    int ny = grid_cy + dy;
                    if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                        if (std::sqrt(dx * dx + dy * dy) <= grid_radius) {
                            grid[nx][ny] = 1;
                        }
                    }
                }
            }
        // finish
    }
};
class AStarPlanner {
public:
    AStarPlanner(int width, int height, double m_min, double m_max, double res) : width_(width), height_(height), map_min_(m_min), map_max_(m_max), grid_resolution_(res), grid_map_(width, height, map_min_, map_max_, grid_resolution_), num_of_obs_(0) {

    }
    AStarPlanner(){};

    void setObstacle(double cx, double cy, double radius) {
        num_of_obs_++;
        grid_map_.markObstacle(cx, cy, radius);
    }

    void printGridMap(){
        for(int i = 0; i < width_; i++){
            for(int j = 0; j < height_; j++){
                std::cout<<grid_map_.grid[i][j]<<" ";
            }
            std::cout<<std::endl;
        }
        std::cout<<"num of obstacles: "<<num_of_obs_<<std::endl;
    }

    double calDist(const Eigen::Vector2d &pos){
        int grid_x = std::round((pos.x() - map_min_) / grid_resolution_);
        int grid_y = std::round((pos.y() - map_min_) / grid_resolution_);
        double min_dist = std::numeric_limits<double>::max();

        for (int i = 0; i < width_; ++i) {
            for (int j = 0; j < height_; ++j) {
                if (grid_map_.grid[i][j] == 1) {
                    double dist = std::sqrt(std::pow(grid_x - i, 2) + std::pow(grid_y - j, 2)) * grid_resolution_;
                    if (dist < min_dist) {
                        min_dist = dist;
                    }
                }
            }
        }

        return min_dist;
    }
    Eigen::Vector2d indexToPosition(int x, int y) {
        double wx = x * grid_resolution_ + map_min_;
        double wy = y * grid_resolution_ + map_min_;
        return Eigen::Vector2d(wx, wy);
    }

    /**
     * @brief Finds a path from the start position to the goal position using the A* algorithm.
     * 
     * @param start The starting position in world coordinates.
     * @param goal The goal position in world coordinates.
     * @return std::vector<Eigen::Vector2d> The path from start to goal as a vector of points in world coordinates.
     *         If no path is found, returns an empty vector.
     * 
     * This function implements the A* pathfinding algorithm. It converts the start and goal positions
     * from world coordinates to grid coordinates, then uses a priority queue (open_list) to explore
     * the grid. The function maintains a closed list to keep track of visited nodes and avoids obstacles.
     * 
     * The algorithm proceeds as follows:
     * 1. Initialize the open list with the start node.
     * 2. While the open list is not empty:
     *    a. Extract the node with the lowest f-cost from the open list.
     *    b. If this node is the goal, reconstruct and return the path.
     *    c. Mark the current node as visited by adding it to the closed list.
     *    d. For each neighbor of the current node:
     *       i. Skip the neighbor if it is in the closed list or is an obstacle.
     *       ii. Calculate the tentative g-cost for the neighbor.
     *       iii. If the neighbor is not in the open list, add it with the calculated costs.
     *       iv. If the neighbor is in the open list and the new g-cost is lower, update the costs and parent.
     * 3. If the open list is exhausted without finding the goal, return an empty path.
     */
    std::vector<Eigen::Vector2d> findPath(Eigen::Vector2d start, Eigen::Vector2d goal) {
        if(num_of_obs_ == 0){
            return {};
        }
        // 起点和终点转换为网格坐标
        auto gridStart = worldToGrid(start);
        auto gridGoal = worldToGrid(goal);

        // 开放列表和关闭列表
        std::priority_queue<std::shared_ptr<Node>, std::vector<std::shared_ptr<Node>>, cmp> open_list;
        std::vector<std::vector<bool>> closed_list(width_, std::vector<bool>(height_, false));

        // 起点加入开放列表
        open_list.push(std::make_shared<Node>(Node(gridStart.first, gridStart.second, 0.0, heuristic(gridStart, gridGoal))));
        // Step 3： 实现 A* 算法，搜索结束调用 reconstructPath 返回路径

            // 样例路径，用于给出路径形式，实现 A* 算法时请删除
                // std::vector<Eigen::Vector2d> path;
                // int num_points = 100; // 生成路径上的点数
                // for (int i = 0; i <= num_points; ++i) {
                //     double t = static_cast<double>(i) / num_points;
                //     Eigen::Vector2d point = start + t * (goal - start);
                //     path.push_back(point);
                // }
                // return path;
            // 注释结束
            // your code

        while (!open_list.empty()) {
            // 获取开放列表中 f 值最小的节点
            auto current = open_list.top();
            open_list.pop();

            // 如果当前节点是目标节点，回溯路径
            if (abs(current->x - gridGoal.first) < 1 && abs(current->y - gridGoal.second) < 1) {
            return reconstructPath(current);
            }

            // 将当前节点加入关闭列表
            closed_list[current->x][current->y] = true;

            // 获取当前节点的所有邻居节点
            for (const auto& neighbor : getNeighbors(*current)) {
            if (closed_list[neighbor.x][neighbor.y] || grid_map_.grid[neighbor.x][neighbor.y] == 1) {
                continue; // 跳过已处理的节点或障碍物
            }

            double tentative_g_cost = current->g_cost + distance(*current, neighbor);
            bool in_open_list = false;

            // 检查邻居节点是否在开放列表中
            std::priority_queue<std::shared_ptr<Node>, std::vector<std::shared_ptr<Node>>, cmp> temp_queue = open_list;
            while (!temp_queue.empty()) {
                auto node = temp_queue.top();
                temp_queue.pop();
                if (node->x == neighbor.x && node->y == neighbor.y) {
                    in_open_list = true;
                    if (tentative_g_cost < node->g_cost) {
                        node->g_cost = tentative_g_cost;
                        node->parent = current;
                    }
                    break;
                }
            }

            // 如果邻居节点不在开放列表中，加入开放列表
            if (!in_open_list) {
                open_list.push(std::make_shared<Node>(neighbor.x, neighbor.y, tentative_g_cost, heuristic({neighbor.x, neighbor.y}, gridGoal), current));
            }
            }
        }

        // finish

        // 如果没有找到路径，返回空路径
        return {};
    }
    void reset(){
        num_of_obs_ = 0;
        grid_map_.grid = std::vector<std::vector<int>>(width_, std::vector<int>(height_, 0));
    }
private:

    // 计算启发式代价（使用欧几里得距离）
    double heuristic(const std::pair<int, int>& from, const std::pair<int, int>& to) {
        return 1.001 * std::sqrt(std::pow(from.first - to.first, 2) + std::pow(from.second - to.second, 2));
    }

    // 计算两节点之间的距离（用于邻居代价计算）
    double distance(const Node& a, const Node& b) {
        return std::sqrt(std::pow(a.x - b.x, 2) + std::pow(a.y - b.y, 2));
    }

    // 从世界坐标转换到栅格坐标
    std::pair<int, int> worldToGrid(const Eigen::Vector2d& position) {
        int x = std::round((position.x() - map_min_) / grid_resolution_);
        int y = std::round((position.y() - map_min_) / grid_resolution_);
        return {x, y};
    }

    // 从栅格坐标转换到世界坐标（主要用于路径结果显示）
    Eigen::Vector2d gridToWorld(int x, int y) {
        double wx = x * grid_resolution_ + map_min_;
        double wy = y * grid_resolution_ + map_min_;
        return Eigen::Vector2d(wx, wy);
    }

    // 获取当前节点的所有邻居节点
    std::vector<Node> getNeighbors(const Node& current) {
        std::vector<Node> neighbors;

        // 八连通邻居
        std::vector<std::pair<int, int>> directions = {
                {1, 0}, {0, 1}, {-1, 0}, {0, -1}, {1, 1}, {1, -1}, {-1, 1}, {-1, -1}};
        for (const auto& dir : directions) {
            // Step 2: 根据当前节点和方向计算邻居节点的坐标，并将其加入 neighbors

                int nx = current.x + dir.first;
                int ny = current.y + dir.second;
                if (nx >= 0 && nx < width_ && ny >= 0 && ny < height_) {
                    neighbors.emplace_back(nx, ny, 0.0, 0.0); // g_cost and h_cost will be updated later
                }

            // finish
        }

        return neighbors;
    }

    // 回溯路径
    std::vector<Eigen::Vector2d> reconstructPath(std::shared_ptr<Node> node) {
        std::vector<Eigen::Vector2d> path;
        while (node) {
            path.push_back(gridToWorld(node->x, node->y));
            node = node->parent;
        }
        std::reverse(path.begin(), path.end());
        reset();
        return path;
    }

    // 地图数据
    int width_, height_;
    double map_min_, map_max_, grid_resolution_;
    GridMap grid_map_; // 栅格地图，0: 空闲，1: 障碍物
    int num_of_obs_;
};