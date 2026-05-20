#!/bin/bash
# 基准策略快速运行脚本

echo "=========================================="
echo " 基准策略 vs TRAST 对比运行脚本"
echo "=========================================="

# 设置颜色
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 检查Python环境
if ! command -v python &> /dev/null; then
    echo "错误: 未找到Python"
    exit 1
fi

# 检查数据目录
if [ ! -d "data/CSI" ]; then
    echo "错误: 未找到数据目录 data/CSI"
    exit 1
fi

# 检查TRAST结果
if [ ! -d "stage2_policy/results" ]; then
    echo -e "${YELLOW}警告: 未找到TRAST结果目录 stage2_policy/results${NC}"
    echo "将只运行基准策略，不会与TRAST对比"
fi

# 默认参数
MODE="test"
ARCH_TYPE="MHA_RoPE_MoE"

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --mode)
            MODE="$2"
            shift 2
            ;;
        --arch_type)
            ARCH_TYPE="$2"
            shift 2
            ;;
        --help)
            echo "使用方法: ./run_benchmarks.sh [选项]"
            echo ""
            echo "选项:"
            echo "  --mode MODE         数据集模式 (train/eval/test, 默认: test)"
            echo "  --arch_type ARCH    TRAST架构类型 (base/MHA_RoPE_MoE/MQA_RoPE_MoE/GQA_RoPE_MoE, 默认: MHA_RoPE_MoE)"
            echo "  --help              显示此帮助信息"
            exit 0
            ;;
        *)
            echo "未知选项: $1"
            echo "使用 --help 查看帮助信息"
            exit 1
            ;;
    esac
done

# 显示运行配置
echo ""
echo "运行配置:"
echo "  数据集模式: $MODE"
echo "  TRAST架构:  $ARCH_TYPE"
echo ""

# 确认运行
read -p "确认运行? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "已取消运行"
    exit 0
fi

# 运行基准策略对比
echo ""
echo "开始运行基准策略对比..."
echo "=========================================="

python run_benchmark_comparison.py --mode "$MODE" --arch_type "$ARCH_TYPE"

# 检查运行结果
if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}✅ 基准策略对比运行完成！${NC}"
    echo ""
    echo "结果保存在: benchmarks/results/${MODE}_${ARCH_TYPE}/"
    echo ""
    echo "生成的文件:"
    ls -la "benchmarks/results/${MODE}_${ARCH_TYPE}/" 2>/dev/null || echo "  (结果目录不存在)"
else
    echo ""
    echo -e "${YELLOW}⚠️  运行过程中出现错误${NC}"
    exit 1
fi