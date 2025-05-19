cmake_minimum_required(VERSION 3.8)
project(custom_interfaces)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Encuentra las dependencias necesarias
find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)

# ✅ Encuentra Python3 y NumPy
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
execute_process(
  COMMAND "${Python3_EXECUTABLE}" -c "import numpy; print(numpy.get_include())"
  OUTPUT_VARIABLE NumPy_INCLUDE_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
include_directories(${NumPy_INCLUDE_DIR})

# Define las interfaces que se deben generar
rosidl_generate_interfaces(${PROJECT_NAME}
  "srv/SetProcessBool.srv"
  "srv/SwitchPublisher.srv"
  DEPENDENCIES std_msgs geometry_msgs
)

# Solo si haces testing
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
