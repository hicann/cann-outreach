#!/bin/bash
# sparse_gemm 运行脚本
# 用于编译、生成测试数据、运行算子并验证结果

set -e  # 遇到错误立即退出

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 打印带颜色的消息
print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 检查命令是否存在
check_command() {
    if ! command -v "$1" &> /dev/null; then
        print_error "Command '$1' not found. Please install it first."
        exit 1
    fi
}

# 检查环境变量
check_env() {
    if [ -z "$ASCEND_HOME_PATH" ]; then
        print_error "ASCEND_HOME_PATH is not set. Please set it first."
        exit 1
    fi
    
    if [ ! -d "$ASCEND_HOME_PATH" ]; then
        print_error "ASCEND_HOME_PATH directory does not exist: $ASCEND_HOME_PATH"
        exit 1
    fi
}

# 清理旧文件
clean() {
    print_info "Cleaning old files..."
    rm -rf build
    rm -rf input
    rm -rf output
}

# 生成测试数据
generate_data() {
    print_info "Generating test data..."
    
    # 检查 Python 是否可用
    check_command python3
    
    # 创建目录
    mkdir -p input
    mkdir -p output
    
    # 运行数据生成脚本
    python3 scripts/gen_data.py \
        --batch 2 \
        --m 128 \
        --n 128 \
        --k 128 \
        --output_dir ./input \
        --seed 42
    
    if [ $? -ne 0 ]; then
        print_error "Failed to generate test data"
        exit 1
    fi
    
    print_info "Test data generated successfully"
}

# 编译算子
build() {
    print_info "Building sparse_gemm..."
    
    # 检查编译器
    check_command cmake
    check_command make
    
    # 创建构建目录
    mkdir -p build
    cd build
    
    # 运行 CMake
    print_info "Running CMake..."
    cmake ..
    
    if [ $? -ne 0 ]; then
        print_error "CMake failed"
        exit 1
    fi
    
    # 编译
    print_info "Compiling..."
    make -j$(nproc)
    
    if [ $? -ne 0 ]; then
        print_error "Compilation failed"
        exit 1
    fi
    
    cd ..
    print_info "Build completed successfully"
}

# 运行算子
run() {
    print_info "Running sparse_gemm..."
    
    # 检查可执行文件是否存在
    if [ ! -f "build/sparse_gemm" ]; then
        print_error "Executable not found: build/sparse_gemm"
        print_info "Please run './run.sh build' first"
        exit 1
    fi
    
    # 运行算子（从项目根目录运行，因为输入文件路径是相对路径）
    ./build/sparse_gemm
    
    if [ $? -ne 0 ]; then
        print_error "Execution failed"
        exit 1
    fi
    
    print_info "Execution completed successfully"
}

# 验证结果
verify() {
    print_info "Verifying result..."
    
    # 检查 Python 是否可用
    check_command python3
    
    # 检查输出文件是否存在
    if [ ! -f "output/output.bin" ]; then
        print_error "Output file not found: output/output.bin"
        exit 1
    fi
    
    # 运行验证脚本
    python3 scripts/verify_result.py \
        --output ./output/output.bin \
        --golden ./input/golden.bin \
        --input_dir ./input \
        --atol 1e-6 \
        --rtol 1e-6 \
        --verbose
    
    if [ $? -ne 0 ]; then
        print_error "Verification failed"
        exit 1
    fi
    
    print_info "Verification completed successfully"
}

# 显示帮助信息
show_help() {
    echo "Usage: $0 [OPTION]"
    echo ""
    echo "Options:"
    echo "  clean       Clean old build and data files"
    echo "  data        Generate test data"
    echo "  build       Build the operator"
    echo "  run         Run the operator"
    echo "  verify      Verify the result"
    echo "  all         Run all steps (clean, data, build, run, verify)"
    echo "  --skip-build  Skip build step (for code review phase)"
    echo "  --help      Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 all          # Run all steps"
    echo "  $0 build        # Only build"
    echo "  $0 run          # Only run (requires build first)"
    echo "  $0 verify       # Only verify (requires run first)"
    echo "  $0 --skip-build # Skip build, run data/run/verify"
}

# 主函数
main() {
    # 检查环境
    check_env
    
    # 解析参数
    case "${1:-all}" in
        clean)
            clean
            ;;
        data)
            generate_data
            ;;
        build)
            build
            ;;
        run)
            run
            ;;
        verify)
            verify
            ;;
        all)
            clean
            generate_data
            build
            run
            verify
            ;;
        --skip-build)
            clean
            generate_data
            run
            verify
            ;;
        --help|-h)
            show_help
            ;;
        *)
            print_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
}

# 运行主函数
main "$@"
