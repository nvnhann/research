package com.udacity.catpoint.security.service;

import com.udacity.catpoint.image.FakeImageService;
import com.udacity.catpoint.security.application.StatusListener;
import com.udacity.catpoint.security.data.*;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.junit.jupiter.params.ParameterizedTest;

import static org.junit.jupiter.api.Assertions.assertFalse;

import org.junit.jupiter.params.provider.EnumSource;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import java.awt.image.BufferedImage;
import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import static org.mockito.Mockito.*;

@ExtendWith(MockitoExtension.class)
public class SecurityServiceTest {
    @Mock
    private FakeImageService imageService;

    @Mock
    private SecurityRepository securityRepository;

    @Mock
    private StatusListener statusListener;

    private SecurityService securityService;

    private Sensor sensor;

    private final String randomString =  UUID.randomUUID().toString();

    @BeforeEach
    void __init() {
        securityService = new SecurityService(securityRepository, imageService);
        sensor = new Sensor(randomString, SensorType.DOOR);
    }

    // test 1: If alarm is armed and a sensor becomes activated, put the system into pending alarm status.
    @Test
    void alarmIsArmedAndSensorBecomesActivated_changeStatusIsPending(){
        when(securityRepository.getArmingStatus()).thenReturn(ArmingStatus.ARMED_HOME);
        when(securityRepository.getAlarmStatus()).thenReturn(AlarmStatus.NO_ALARM);
        securityService.changeSensorActivationStatus(sensor, true);
        verify(securityRepository).setAlarmStatus(AlarmStatus.PENDING_ALARM);
    }

    // test 2: If alarm is armed and a sensor becomes activated and the system is already pending alarm, set the alarm status to alarm.
    @Test
    void alarmIsArmedAndSensorBecomesActivatedSystemIsPending_changeStatusAlarm(){
        when(securityRepository.getArmingStatus()).thenReturn(ArmingStatus.ARMED_HOME);
        when(securityRepository.getAlarmStatus()).thenReturn(AlarmStatus.PENDING_ALARM);
        securityService.changeSensorActivationStatus(sensor, true);
        verify(securityRepository).setAlarmStatus(AlarmStatus.ALARM);
    }

    // test 3: If pending alarm and all sensors are inactive, return to no alarm state.
    @Test
    void alarmIsPendingAndSensorInactivate_returnNoAlarm(){
        when(securityRepository.getAlarmStatus()).thenReturn(AlarmStatus.PENDING_ALARM);
        when(securityRepository.getArmingStatus()).thenReturn(ArmingStatus.ARMED_HOME);
        sensor.setActive(false);
        securityService.changeSensorActivationStatus(sensor, true);
        securityService.changeSensorActivationStatus(sensor, false);
        verify(securityRepository).setAlarmStatus(AlarmStatus.NO_ALARM);
    }

    // test 4: If alarm is active, change in sensor state should not affect the alarm state.
    @Test
    void alarmIsActive_changeSensorShouldNotAffectAlarmState(){
        when(securityRepository.getAlarmStatus()).thenReturn(AlarmStatus.ALARM);
        sensor.setActive(false);
        securityService.changeSensorActivationStatus(sensor, true);
        verify(securityRepository, never()).setAlarmStatus(any(AlarmStatus.class));
    }

    // test 5: If a sensor is activated while already active and the system is in pending state, change it to alarm state.
    @Test
    void sensorActivatedWhileActiveAndPendingAlarm_changeStatusToAlarm(){
        when(securityRepository.getAlarmStatus()).thenReturn(AlarmStatus.PENDING_ALARM);
        sensor.setActive(true);
        securityService.changeSensorActivationStatus(sensor, true);
        verify(securityRepository).setAlarmStatus(AlarmStatus.ALARM);
    }

    // test 6: If a sensor is deactivated while already inactive, make no changes to the alarm state.
    @Test
    void sensorDeactivatedWhileInactive_noChangesToAlarmState(){
        when(securityRepository.getAlarmStatus()).thenReturn(AlarmStatus.PENDING_ALARM);
        sensor.setActive(false);
        securityService.changeSensorActivationStatus(sensor, false);
        verify(securityRepository, never()).setAlarmStatus(any(AlarmStatus.class));
    }

    // test 7: If the image service identifies an image containing a cat while the system is armed-home, put the system into alarm status.
    @Test
    void imageServiceIdentifiesContainCatWhileAlarmArmedHome_changeStatusToAlarm(){
        when(imageService.imageContainsCat(any(), anyFloat())).thenReturn(true);
        when(securityService.getArmingStatus()).thenReturn(ArmingStatus.ARMED_HOME);
        securityService.processImage(mock(BufferedImage.class));
        //securityService.processImage(new BufferedImage(255,255,1));
        verify(securityRepository).setAlarmStatus(AlarmStatus.ALARM);
    }

    // test 8: If the image service identifies an image that does not contain a cat, change the status to no alarm as long as the sensors are not active.
    @Test
    void imageServiceIdentifiesNotContainCat_changeStatusToNoAlarmAsLongSensorsNotActive(){
        when(imageService.imageContainsCat(any(), anyFloat())).thenReturn(false);
        sensor.setActive(false);
        //securityService.processImage(new BufferedImage(255,255,1));
        securityService.processImage(mock(BufferedImage.class));

        verify(securityRepository).setAlarmStatus(AlarmStatus.NO_ALARM);
    }

    // test 9: If the system is disarmed, set the status to no alarm.
    @Test
    void systemDisarmed_setNoAlarmState(){
        securityService.setArmingStatus(ArmingStatus.DISARMED);
        verify(securityRepository).setAlarmStatus(AlarmStatus.NO_ALARM);
    }

    // test 10: If the system is armed, reset all sensors to inactive.
    @ParameterizedTest
    @EnumSource(value = ArmingStatus.class, names = {"ARMED_AWAY", "ARMED_HOME"})
    void systemArmed_resetSensorsToInactive(ArmingStatus status) {

        Set<Sensor> sensors = Stream.of(
                new Sensor("DOOR", SensorType.DOOR),
                new Sensor("WINDOW", SensorType.WINDOW),
                new Sensor("MOTION", SensorType.MOTION)
        ).collect(Collectors.toSet());

        when(securityRepository.getAlarmStatus()).thenReturn(AlarmStatus.PENDING_ALARM);
        when(securityRepository.getSensors()).thenReturn(sensors);

        sensors.forEach(sensor -> sensor.setActive(true));
        securityService.setArmingStatus(status);

        sensors.forEach(sensor -> assertFalse(sensor.getActive()));
    }

    // test 11: If the system is armed-home while the camera shows a cat, set the alarm status to alarm.
    @Test
    void systemIsArmedHomeWhileCameraShowCat_changeStatusToAlarm(){
        when(imageService.imageContainsCat(any(), anyFloat())).thenReturn(true);
        when(securityService.getArmingStatus()).thenReturn(ArmingStatus.ARMED_HOME);
        securityService.processImage(mock(BufferedImage.class));
        //securityService.processImage(new BufferedImage(255,255,1));
        verify(securityRepository).setAlarmStatus(AlarmStatus.ALARM);
    }

    // test add and remove sensor, listener
    @Test
    void addAndRemoveSensorAndListener(){
        securityService.addSensor(sensor);
        securityService.removeSensor(sensor);
        securityService.addStatusListener(statusListener);
        securityService.removeStatusListener(statusListener);
    }

    @Test
    void systemArmedHomeAndImageContainsCat_setsAlarmToAlarm() {
        when(imageService.imageContainsCat(any(), anyFloat())).thenReturn(true);
        when(securityRepository.getArmingStatus()).thenReturn(ArmingStatus.ARMED_HOME);
        securityService.processImage(new BufferedImage(255,255,1));
        verify(securityRepository).setAlarmStatus(AlarmStatus.ALARM);
    }

    @Test
    void detectionTrueAndSystemArmedHome_setsAlarmToAlarm() {
        securityService.setDetection(true);
        securityService.setArmingStatus(ArmingStatus.ARMED_HOME);
        verify(securityRepository).setAlarmStatus(AlarmStatus.ALARM);
    }

    @Test
    void alarmStatusIsAlarmOnSensorDeactivated_setsAlarmToPendingAlarm() {
        when(securityRepository.getAlarmStatus()).thenReturn(AlarmStatus.ALARM);
        securityService.handleSensorDeactivated();
        verify(securityRepository).setAlarmStatus(AlarmStatus.PENDING_ALARM);
    }

}
