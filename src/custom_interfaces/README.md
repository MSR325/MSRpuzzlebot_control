# ✅ Verificar que un paquete ROS 2 compile correctamente servicios `.srv`

Este checklist asegura que un paquete con servicios personalizados compile correctamente usando `colcon build` (sin `--symlink-install`).

---

## 1. Verificar estructura del paquete

Asegúrate de que tu paquete tenga la siguiente estructura:

```
custom_interfaces/
├── CMakeLists.txt
├── package.xml
└── srv/
    ├── SetProcessBool.srv
    └── SwitchPublisher.srv
```

---

## 2. `CMakeLists.txt`

Debe contener:

```cmake
cmake_minimum_required(VERSION 3.8)
project(custom_interfaces)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "srv/SetProcessBool.srv"
  "srv/SwitchPublisher.srv"
  DEPENDENCIES std_msgs geometry_msgs
)

ament_package()
```

---

## 3. `package.xml`

Debe incluir las siguientes dependencias:

```xml
<buildtool_depend>ament_cmake</buildtool_depend>
<buildtool_depend>rosidl_default_generators</buildtool_depend>

<build_depend>rosidl_default_generators</build_depend>
<exec_depend>rosidl_default_runtime</exec_depend>

<build_depend>std_msgs</build_depend>
<exec_depend>std_msgs</exec_depend>

<build_depend>geometry_msgs</build_depend>
<exec_depend>geometry_msgs</exec_depend>

<member_of_group>rosidl_interface_packages</member_of_group>
```

---

## 4. Limpiar el workspace antes del build

```bash
rm -rf build/ install/ log/
```

---

## 5. Compilar con build normal

```bash
colcon build
source install/setup.bash
```

---

## 6. Verificar que el servicio se haya generado

```bash
ros2 interface show custom_interfaces/srv/SwitchPublisher
ros2 service list
```

---

Con estos pasos, tus servicios `.srv` estarán correctamente generados y listos para ser usados desde nodos o terminal.
