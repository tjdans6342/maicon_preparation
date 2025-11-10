#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy


class PIDController:
    """
    ✅ PID 제어 클래스
    - 오차(error)를 입력받아 제어 출력값을 반환
    - P: 즉각적 반응 / I: 누적 오차 보정 / D: 변화율 억제
    """

    def __init__(self, kp=0.65, ki=0.001, kd=0.01, integral_limit=2.0):
        """
        Parameters
        ----------
        kp : float
            비례 게인 (Proportional)
        ki : float
            적분 게인 (Integral)
        kd : float
            미분 게인 (Derivative)
        integral_limit : float or None
            적분항 누적 제한 (과적분 방지)
        """
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral_limit = integral_limit

        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_time = None

    # -------------------------------------------------------
    #  오차 업데이트 → PID 출력 반환
    # -------------------------------------------------------
    def update(self, error, current_time=None):
        """
        Parameters
        ----------
        error : float
            현재 오차 값 (목표 - 실제)
        current_time : float, optional
            현재 시각 (rospy.get_time() 또는 time.time())

        Returns
        -------
        float : 제어 출력값
        """
        if current_time is None:
            current_time = rospy.get_time()

        if self.prev_time is None:
            dt = 0.0
        else:
            dt = max(current_time - self.prev_time, 0.0)

        # 미분 항
        derivative = (error - self.prev_error) / dt if dt > 0 else 0.0

        # 적분 항 누적
        self.integral += error * dt
        if self.integral_limit is not None:
            self.integral = max(min(self.integral, self.integral_limit), -self.integral_limit)

        # PID 출력 계산
        output = (self.kp * error) + (self.ki * self.integral) + (self.kd * derivative)

        # 상태 갱신
        self.prev_error = error
        self.prev_time = current_time
        return output

    # -------------------------------------------------------
    #  내부 상태 초기화
    # -------------------------------------------------------
    def reset(self):
        """적분항 및 이전 오차 초기화"""
        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_time = None
        rospy.loginfo("🔄 PIDController reset")

    # -------------------------------------------------------
    #  디버깅용 문자열 출력
    # -------------------------------------------------------
    def __repr__(self):
        return "<PIDController kp={}, ki={}, kd={}>".format(self.kp, self.ki, self.kd)
