import RPi.GPIO as GPIO
import time
import sys

GPIO_DIR_PIN_PUMP = 24  # BCM pin for L298N IN2 (Direction)
GPIO_PWM_PIN_PUMP = 12  # BCM pin for L298N ENA (Speed Control / Enable)

PWM_FREQUENCY_PUMP = 100  # Hz
SPEED_PUMP = 50   # % Duty Cycle (0-100)

try:
    # Use BCM numbering mode
    GPIO.setmode(GPIO.BCM)

    # Configure pins as outputs
    GPIO.setup(GPIO_DIR_PIN_PUMP, GPIO.OUT)
    GPIO.setup(GPIO_PWM_PIN_PUMP, GPIO.OUT)

    # Initialize PWM on the enable pin
    pump_pwm = GPIO.PWM(GPIO_PWM_PIN_PUMP, PWM_FREQUENCY_PUMP)

    # Start the motor with the initial speed
    # The direction (IN1) is set to HIGH for one direction
    GPIO.output(GPIO_DIR_PIN_PUMP, GPIO.HIGH)
    pump_pwm.start(SPEED_PUMP)
    
    print(f"Motor running at {SPEED_PUMP}% speed (Direction HIGH).")
    print("Press Ctrl+C to stop sooner.")

    # Run for a set duration
    time.sleep(3) 
    
    # Stop PWM and cleanup
    pump_pwm.stop()
    print("\nMotor stopped.")

except KeyboardInterrupt:
    print("\nScript stopped by user.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
    sys.exit(1)
finally:
    # 8. Cleanup: Critical step to reset pins on exit
    GPIO.cleanup()
    print("GPIO cleanup complete. Pins reset to default state.")