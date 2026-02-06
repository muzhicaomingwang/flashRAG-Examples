#!/bin/bash
# 安全运行优化版HippoRAG实验
# 包含完整的检查、运行、评估流程

set -e  # 遇到错误立即退出

# 颜色定义
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# 打印带颜色的消息
print_info() {
    echo -e "${GREEN}$1${NC}"
}

print_warning() {
    echo -e "${YELLOW}$1${NC}"
}

print_error() {
    echo -e "${RED}$1${NC}"
}

# 打印横幅
print_banner() {
    echo "========================================================================"
    echo "$1"
    echo "========================================================================"
}

# 主流程开始
print_banner "HippoRAG优化版实验 - 安全执行流程"

# Step 1: 激活虚拟环境
print_info "\n📦 Step 1: 激活虚拟环境"

if [ -d "venv" ]; then
    source venv/bin/activate
    print_info "✅ 虚拟环境已激活"
else
    print_error "✗ 虚拟环境不存在，请先运行: python -m venv venv"
    exit 1
fi

# Step 2: 环境检查
print_info "\n🔍 Step 2: 环境检查"

if python scripts/check_environment.py; then
    print_info "✅ 环境检查通过"
else
    print_error "✗ 环境检查失败，请先修复问题"
    exit 1
fi

# Step 3: 检查是否有checkpoint（断点续传）
print_info "\n💾 Step 3: 检查checkpoint"

if [ -f "results/checkpoint_baseline.json" ]; then
    baseline_checkpoint=$(python -c "import json; print(len(json.load(open('results/checkpoint_baseline.json'))))")
    print_warning "⚠️  发现Baseline checkpoint: 已完成 $baseline_checkpoint/500"
    read -p "是否从checkpoint恢复？(y/n，默认y): " resume_baseline
    resume_baseline=${resume_baseline:-y}

    if [ "$resume_baseline" != "y" ]; then
        print_warning "删除Baseline checkpoint，重新开始..."
        rm results/checkpoint_baseline.json
    fi
fi

if [ -f "results/checkpoint_hipporag.json" ]; then
    hipporag_checkpoint=$(python -c "import json; print(len(json.load(open('results/checkpoint_hipporag.json'))))")
    print_warning "⚠️  发现HippoRAG checkpoint: 已完成 $hipporag_checkpoint/500"
    read -p "是否从checkpoint恢复？(y/n，默认y): " resume_hipporag
    resume_hipporag=${resume_hipporag:-y}

    if [ "$resume_hipporag" != "y" ]; then
        print_warning "删除HippoRAG checkpoint，重新开始..."
        rm results/checkpoint_hipporag.json
    fi
fi

# Step 4: 运行优化实验
print_info "\n🚀 Step 4: 运行优化实验"
print_info "预计时间: 60分钟"
print_info "预计成本: $2.02"
print_info ""
print_warning "提示: 你可以随时按Ctrl+C中断，实验会保存checkpoint并可以恢复"
print_info ""

read -p "按Enter继续，或Ctrl+C取消..." confirm

# 记录开始时间
start_time=$(date +%s)

# 运行实验
if python scripts/run_optimized_experiment_safe.py; then
    print_info "\n✅ 实验运行成功！"
else
    print_error "\n✗ 实验运行失败"
    print_warning "检查checkpoint文件，可以重新运行此脚本恢复进度"
    exit 1
fi

# 计算运行时间
end_time=$(date +%s)
duration=$((end_time - start_time))
duration_min=$((duration / 60))
print_info "⏱️  总用时: ${duration_min}分钟"

# Step 5: 评估结果
print_info "\n📊 Step 5: 评估结果"

if python scripts/evaluate_results_simple.py; then
    print_info "✅ 评估完成"
else
    print_error "✗ 评估失败"
    exit 1
fi

# Step 6: 显示结果
print_info "\n📋 Step 6: 实验结果"
print_banner "对比报告"

if [ -f "results/comparison_report.md" ]; then
    cat results/comparison_report.md
else
    print_error "✗ 报告文件不存在"
    exit 1
fi

# 完成
print_banner "✅ 实验全部完成！"

print_info "\n📁 结果文件:"
print_info "  - results/comparison_report.md       (对比报告)"
print_info "  - results/evaluation_metrics.json    (详细指标)"
print_info "  - results/baseline_full/predictions.json  (Baseline预测)"
print_info "  - results/hipporag_full/predictions.json  (HippoRAG预测)"

print_info "\n💡 下一步:"
print_info "  - 查看详细报告: cat results/comparison_report.md"
print_info "  - 分析预测结果: python scripts/analyze_predictions.py"
print_info "  - 如果结果不理想，可以尝试调整参数重新运行"

print_info "\n🎉 感谢使用HippoRAG实验框架！"
