import cv2
import numpy as np
import time
# import math

#… (이전 설정과 Machine 클래스 정의는 그대로)
pos = [0,0,0]
angOrig = 206.662752199
speed = [0, 0, 0] 
speedPrev = [0, 0, 0] 
ks = 20
Xoffset = 0 # 화면 중심 맞추기
Yoffset = 0
kp = 4E-4, ki = 2E-6, kd = 7E-3
error =  [0, 0], errorPrev, integr =  [0, 0], deriv =  [0, 0], out = [0, 0]
timeI = 0 
angToStep = 200 / 360
detected = 0

# Leg identifiers
A, B, C = 0, 1, 2
stepperA_maxspeed = 0
stepperB_maxspeed = 0
stepperC_maxspeed = 0
stepperA_Acceleration = 0
stepperB_Acceleration = 0
stepperC_Acceleration = 0




# 1) 웹캠 열기 (C270이 두 번째 장치라면 index=1)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)
if not cap.isOpened():
    print("웹캠을 열 수 없습니다.")
    exit()
prev_center = None  # 이전 프레임의 중심 좌표 저장용


# millis() 함수처럼 동작하는 함수
def millis():
    return int(round(time.time() * 1000))  # 현재 시간을 밀리초로 반환


def constrain(val, min_val, max_val):
    return max(min_val, min(val, max_val))


class Machine:
    def __init__(self, d: float, e: float, f: float, g: float):
        """
        :param d: distance from the center of the base to any of its corners
        :param e: distance from the center of the platform to any of its corners
        :param f: length of link #1
        :param g: length of link #2
        """
        self.d, self.e, self.f, self.g = d, e, f, g

    def theta(self, leg: int, hz: float, nx: float, ny: float) -> float:
        """
        Compute the motor angle for leg A/B/C of a 3RPS manipulator.

        :param leg: leg index (A=0, B=1, C=2)
        :param hz: vertical height of the platform
        :param nx: x-component of platform normal before normalization
        :param ny: y-component of platform normal before normalization
        :return: angle (degrees) that the motor must rotate
        """
        # Normalize the normal vector [nx, ny, 1]
        n = np.array([nx, ny, 1.0])
        n /= np.linalg.norm(n)  # now [nxn, nyn, nzn]
        nxn, nyn, nzn = n

        if leg == A:
            denom = (nzn + 1 - nxn**2 +
                     (nxn**4 - 3*nxn**2 * nyn**2) /
                     ((nzn + 1) * (nzn + 1 - nxn**2)))
            y = self.d + (self.e / 2) * (1 - (nxn**2 + 3*nzn**2 + 3*nzn) / denom)
            z = hz + self.e * nyn
            mag = np.hypot(y, z)
            alpha = np.arccos(np.clip(y/mag, -1.0, 1.0))
            β = np.arccos(np.clip((mag**2 + self.f**2 - self.g**2) / (2 * mag * self.f), -1.0, 1.0))
            angle = alpha + β

        elif leg == B:
            x = (np.sqrt(3)/2) * (self.e * (1 - (nxn**2 + np.sqrt(3)*nxn*nyn) / (nzn + 1)) - self.d)
            y = x / np.sqrt(3)
            z = hz - (self.e / 2) * (np.sqrt(3)*nxn + nyn)
            mag = np.linalg.norm([x, y, z])
            alpha = np.arccos(np.clip((np.sqrt(3)*x + y) / (-2 * mag), -1.0, 1.0))
            β = np.arccos(np.clip((mag**2 + self.f**2 - self.g**2) / (2 * mag * self.f), -1.0, 1.0))
            angle = alpha + β

        elif leg == C:
            x = (np.sqrt(3)/2) * (self.d - self.e * (1 - (nxn**2 - np.sqrt(3)*nxn*nyn) / (nzn + 1)))
            y = -x / np.sqrt(3)
            z = hz + (self.e / 2) * (np.sqrt(3)*nxn - nyn)
            mag = np.linalg.norm([x, y, z])
            alpha = np.arccos(np.clip((np.sqrt(3)*x - y) / (2 * mag), -1.0, 1.0))
            β = np.arccos(np.clip((mag**2 + self.f**2 - self.g**2) / (2 * mag * self.f), -1.0, 1.0))
            angle = alpha + β

        else:
            raise ValueError(f"Invalid leg index: {leg}")

        # 라디안→도 변환
        return float(angle * 180.0 / np.pi)
machine = Machine(d=2.0, e=3.125, f=1.75, g=3.669291339)

def moveTo(hz, nx, ny):
    global detected
    global stepperA_maxspeed
    global stepperB_maxspeed
    global stepperC_maxspeed
    global stepperA_Acceleration
    global stepperB_Acceleration
    global stepperC_Acceleration

    if detected:
        for i in range(3):
            pos[i] = round((angOrig - machine.theta(i, hz, nx, ny)) * angToStep) # 이것들 다 sum해서 currentPosition만들기
        stepperA_maxspeed = speed[A]
        stepperB_maxspeed = speed[B]
        stepperC_maxspeed = speed[C]

        stepperA_Acceleration = speed[A] * 30
        stepperB_Acceleration = speed[B] * 30
        stepperC_Acceleration = speed[C] * 30

        #pos*, stepper*_maxspeed, stepper*_Acceleration 시리얼 전송

    else :
        for i in range(3):
            pos[i] = round((angOrig - machine.theta(i, hz, 0, 0)) * angToStep) # 아마 양수. 이것들 다 sum해서 currentPosition만들기
        stepperA_maxspeed = 800
        stepperB_maxspeed = 800
        stepperC_maxspeed = 800

        #pos, maxspeed 시리얼 전송


# --- 1) 공 검출 함수 정의 ---
def detect_ball(frame, lower_hsv=(10,100,100), upper_hsv=(25,255,255), area_thresh=100):
    """
    frame: BGR 이미지
    return: detected(bool), cx(int), cy(int)
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(lower_hsv), np.array(upper_hsv))
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return False, None, None

    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < area_thresh:
        return False, None, None

    M = cv2.moments(largest)
    if M['m00'] == 0:
        return False, None, None

    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])
    return True, cx, cy

# --- 2) PID 함수에 검출 로직 삽입 ---
def PID(setpointX, setpointY, frame):
    global detected, p_x, p_y

    detected, cx, cy = detect_ball(frame)
    if detected:
        detected = 1
        p_x, p_y = cx, cy
        for i in range(2):
            errorPrev[i] = error[i]
            error[i] = (i==0)*(Xoffset - p_x - setpointX) + (i==1)*(Yoffset - p_y - setpointY)
            integr[i] += error[i] + errorPrev[i]
            deriv[i] = error[i] - errorPrev[i]
            deriv[i] = 0 if np.isnan(deriv[i]) or np.isinf(deriv[i]) else deriv[i]
            out[i] = kp * error[i] + ki * integr[i] + kd * deriv[i]
            out[i] = constrain(out[i], -0.25, 0.25)

        # 속도 계산
        for i in (A, B, C):
            speedPrev[i] = speed[i]
            speed[i] = currentPosition[i] # speed[i] = (i == A) * stepperA.currentPosition() + (i == B) * stepperB.currentPosition() + (i == C) * stepperC.currentPosition(); 
            speed[i] = abs(speed[i] - pos[i]) * ks
            speed[i] = constrain(speed[i], speedPrev[i] - 200, speedPrev[i] + 200)
            speed[i] = constrain(speed[i], 0, 1000)

        print(f"X OUT = {out[0]:.4f}   Y OUT = {out[1]:.4f}   Speed A: {speed[A]}")
    else:
        time.sleep(10)         
        detected, cx, cy = detect_ball(frame)
        if not detected :  
            print("BALL NOT DETECTED")
            detected = 0


while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 원하는 setpoint값을 넣어서 호출
    PID(setpointX=0, setpointY=0, frame=frame)

    # moveTo 호출
    moveTo(4.25, -out[0], -out[1])

    # 결과 확인용 디스플레이 (선택)
    cv2.putText(frame, f"Detected: {detected}", (10,30),
                cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0,255,0) if detected else (0,0,255), 2)
    cv2.imshow('Ball Balancer', frame)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
