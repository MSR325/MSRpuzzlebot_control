#include <Arduino.h>
#include <micro_ros_arduino.h>
#include <rcl/rcl.h>
#include <rclc/rclc.h>
#include <rclc/executor.h>
#include <std_msgs/msg/float32.h>
#include <std_msgs/msg/float32_multi_array.h>
#include "driver/mcpwm.h"

// --------------------------------------------------------------
// Definiciones para el encoder
// --------------------------------------------------------------
#define ENCODER_A 19
#define ENCODER_B 21

// Ajusta esto a los pulsos por vuelta reales de tu encoder
// (por ejemplo, 240 si realmente son 240 PPR en un solo canal).
#define PULSOS_POR_REV 240  

volatile long motorPosition = 0;  // Contador global del encoder

// ISR para leer encoder (ejemplo sencillo, lee sólo canal A)
void IRAM_ATTR updateMotorPosition() {
  if (digitalRead(ENCODER_B) != digitalRead(ENCODER_A)) {
    motorPosition++;
  } else {
    motorPosition--;
  }
}

// --------------------------------------------------------------
// Pines y configuración del motor/PWM
// --------------------------------------------------------------
#define MOTOR_IN1  5
#define MOTOR_IN2  18
#define PWM_PIN    2
#define PWM_FREQ   500
#define PWM_CHNL   MCPWM_OPR_A

// --------------------------------------------------------------
// Variables de Control PID
// --------------------------------------------------------------
float Kp = 0.02f;
float Ki = 0.2f;
float Kd = 0.00005f; //0.000000008f;

float pid_integral = 0.0f;
float last_error   = 0.0f;

// Setpoint de velocidad (rad/s)
float velocity_setpoint = 0.0f;

// Límite del duty cycle normalizado [-1, 1]
float max_control =  1.0f;
float min_control = -1.0f;

// --------------------------------------------------------------
// Variables para el filtro de primer orden
// --------------------------------------------------------------
float filteredSpeed = 0.0f;  // Velocidad filtrada (rad/s)

// --------------------------------------------------------------
// Funciones Auxiliares
// --------------------------------------------------------------
static float clampf(float val, float min_val, float max_val) {
  if (val > max_val) return max_val;
  if (val < min_val) return min_val;
  return val;
}

float computePID(float setpoint, float measured, float dt) {
  float error = setpoint - measured;
  
  // Anti-windup simple
  if (fabs(setpoint) < 1e-6f) {
    pid_integral = 0.0f;
  } else {
    pid_integral += error * dt;
  }
  
  float derivative = 0.0f;
  if (dt > 1e-6f) {
    derivative = (error - last_error) / dt;
  }
  last_error = error;
  
  float output = (Kp * error) + (Ki * pid_integral) + (Kd * derivative);
  return clampf(output, min_control, max_control);
}

// --------------------------------------------------------------
// Estructuras y objetos ROS 2
// --------------------------------------------------------------
rcl_allocator_t allocator;
rclc_support_t support;
rcl_node_t node;
rclc_executor_t executor;

// Publicadores
rcl_publisher_t encoder_publisher;
std_msgs__msg__Float32 encoder_msg;

rcl_publisher_t control_signal_publisher;
std_msgs__msg__Float32 control_signal_msg;

rcl_publisher_t error_publisher;
std_msgs__msg__Float32 error_msg;

// Suscriptores
rcl_subscription_t vel_subscriber;
std_msgs__msg__Float32 vel_msg;

rcl_subscription_t pid_gains_subscriber;
std_msgs__msg__Float32MultiArray pid_gains_msg;

// Callbacks
void vel_callback(const void *msgin) {
  const std_msgs__msg__Float32 *msg = (const std_msgs__msg__Float32 *)msgin;
  velocity_setpoint = msg->data;
  Serial.print("[SETPOINT] Nuevo set_point: ");
  Serial.println(velocity_setpoint);
}

void pid_gains_callback(const void *msgin) {
  const std_msgs__msg__Float32MultiArray *msg = 
      (const std_msgs__msg__Float32MultiArray *)msgin;
  if (msg->data.size >= 3) {
    Kp = msg->data.data[0];
    Ki = msg->data.data[1];
    Kd = msg->data.data[2];
    Serial.print("[PID] Nuevas ganancias: ");
    Serial.print(Kp); Serial.print(", ");
    Serial.print(Ki); Serial.print(", ");
    Serial.println(Kd);
  } else {
    Serial.print("[PID] Recibido Float32MultiArray pero no hay 3 elementos (size = ");
    Serial.print(msg->data.size);
    Serial.println(").");
  }
}

// --------------------------------------------------------------
// setup()
// --------------------------------------------------------------
unsigned long lastTime = 0;
long lastPositionCount = 0;

void setup() {
  Serial.begin(115200);
  set_microros_transports();
  delay(2000);

  // Encoder
  pinMode(ENCODER_A, INPUT);
  pinMode(ENCODER_B, INPUT);
  attachInterrupt(digitalPinToInterrupt(ENCODER_A), updateMotorPosition, CHANGE);

  // Motor
  pinMode(MOTOR_IN1, OUTPUT);
  pinMode(MOTOR_IN2, OUTPUT);
  digitalWrite(MOTOR_IN1, LOW);
  digitalWrite(MOTOR_IN2, LOW);

  // Configurar MCPWM en ESP32
  mcpwm_gpio_init(MCPWM_UNIT_0, MCPWM0A, PWM_PIN);
  mcpwm_config_t pwm_config = {
    .frequency    = PWM_FREQ,
    .cmpr_a       = 0.0f,
    .cmpr_b       = 0.0f,
    .duty_mode    = MCPWM_DUTY_MODE_0,
    .counter_mode = MCPWM_UP_COUNTER
  };
  mcpwm_init(MCPWM_UNIT_0, MCPWM_TIMER_0, &pwm_config);

  // ROS 2
  allocator = rcl_get_default_allocator();
  rclc_support_init(&support, 0, NULL, &allocator);
  rclc_node_init_default(&node, "motor_control_node", "", &support);

  // Publicadores
  rclc_publisher_init_default(
    &encoder_publisher,
    &node,
    ROSIDL_GET_MSG_TYPE_SUPPORT(std_msgs, msg, Float32),
    "encoder_data"
  );

  rclc_publisher_init_default(
    &control_signal_publisher,
    &node,
    ROSIDL_GET_MSG_TYPE_SUPPORT(std_msgs, msg, Float32),
    "control_signal"
  );

  rclc_publisher_init_default(
    &error_publisher,
    &node,
    ROSIDL_GET_MSG_TYPE_SUPPORT(std_msgs, msg, Float32),
    "error"
  );

  // Suscriptores
  rclc_subscription_init_default(
    &vel_subscriber,
    &node,
    ROSIDL_GET_MSG_TYPE_SUPPORT(std_msgs, msg, Float32),
    "/set_point"
  );

  rclc_subscription_init_default(
    &pid_gains_subscriber,
    &node,
    ROSIDL_GET_MSG_TYPE_SUPPORT(std_msgs, msg, Float32MultiArray),
    "/pid_gains"
  );

  // Asignar memoria al mensaje que usará la suscripción (PID gains)
  pid_gains_msg.data.capacity = 3;
  pid_gains_msg.data.size     = 0;
  pid_gains_msg.data.data     = (float*)malloc(3 * sizeof(float));

  // Executor con 2 suscripciones
  rclc_executor_init(&executor, &support.context, 2, &allocator);
  rclc_executor_add_subscription(&executor, &vel_subscriber, &vel_msg, &vel_callback, ON_NEW_DATA);
  rclc_executor_add_subscription(&executor, &pid_gains_subscriber, &pid_gains_msg, &pid_gains_callback, ON_NEW_DATA);

  lastTime = millis();
  lastPositionCount = motorPosition;

  Serial.println("[Setup] Nodo 'motor_control_node' iniciado.");
}

// --------------------------------------------------------------
// loop()
// --------------------------------------------------------------
void loop() {
  // Procesar callbacks de ROS 2
  rclc_executor_spin_some(&executor, RCL_MS_TO_NS(1));

  // Control + Lectura del encoder cada ~5 ms (se redujo para obtener mayor resolución)
  unsigned long currentTime = millis();
  unsigned long deltaTime = currentTime - lastTime;

  if (deltaTime >= 5) {
    float dt = (float)deltaTime / 1000.0f;  // Convertir dt a segundos

    // Calcular la diferencia de posiciones
    long currentPosition = motorPosition;
    long deltaPosition = currentPosition - lastPositionCount;

    // Pulsos por segundo
    float pulsePerSec = (deltaPosition * 1000.0f) / deltaTime;

    // RPM
    float rpm = (pulsePerSec / PULSOS_POR_REV) * 60.0f;

    // Velocidad angular en rad/s (velocidad medida)
    float measuredSpeed = rpm * (2.0f * PI) / 60.0f;

    // Aplicar filtro de primer orden
    // Ajusta tau según la dinámica deseada; aquí se usa 50 ms (0.05 s)
    const float tau = 0.05f;
    float alpha = dt / (tau + dt);
    filteredSpeed = alpha * measuredSpeed + (1.0f - alpha) * filteredSpeed;

    // Calcular acción de control usando la velocidad filtrada
    float error_val = velocity_setpoint - filteredSpeed;
    float control_output = computePID(velocity_setpoint, filteredSpeed, dt);

    // Aplicar la salida de control al motor
    if (fabs(control_output) < 0.01f) {
      // Valor casi cero => apagar motor
      digitalWrite(MOTOR_IN1, LOW);
      digitalWrite(MOTOR_IN2, LOW);
      mcpwm_set_duty(MCPWM_UNIT_0, MCPWM_TIMER_0, PWM_CHNL, 0);
    } else {
      if (control_output < 0.0f) {
        digitalWrite(MOTOR_IN1, HIGH);
        digitalWrite(MOTOR_IN2, LOW);
      } else {
        digitalWrite(MOTOR_IN1, LOW);
        digitalWrite(MOTOR_IN2, HIGH);
      }
      float duty_cycle = fabs(control_output) * 100.0f;
      mcpwm_set_duty(MCPWM_UNIT_0, MCPWM_TIMER_0, PWM_CHNL, duty_cycle);
    }

    // Publicar la velocidad medida filtrada
    encoder_msg.data = filteredSpeed;
    rcl_publish(&encoder_publisher, &encoder_msg, NULL);

    // Publicar la señal de control
    control_signal_msg.data = control_output;
    rcl_publish(&control_signal_publisher, &control_signal_msg, NULL);

    // Publicar el error
    error_msg.data = error_val;
    rcl_publish(&error_publisher, &error_msg, NULL);

    // Debug: Imprimir en el monitor serie
    Serial.print("[ENCODER] Pulsos totales: ");
    Serial.print(currentPosition);
    Serial.print(" | ΔPulsos: ");
    Serial.print(deltaPosition);
    Serial.print(" | Velocidad (filtrada) [rad/s]: ");
    Serial.print(filteredSpeed);
    Serial.print(" | Setpoint: ");
    Serial.print(velocity_setpoint);
    Serial.print(" | Error: ");
    Serial.print(error_val);
    Serial.print(" | Control: ");
    Serial.println(control_output);

    // Actualizar referencias
    lastPositionCount = currentPosition;
    lastTime = currentTime;
  }

  delay(1);
}
