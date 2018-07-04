import java.io.*;
import lejos.nxt.*;
import lejos.nxt.comm.*;
import lejos.robotics.navigation.DifferentialPilot;

public class Localization {
  private static DataInputStream in;
  private static DataOutputStream out;
  private static USBConnection usb;

  private static DifferentialPilot pilot;
  private static UltrasonicSensor sonar;

  private static final int NOOP = 0x00;
  private static final int SEND = 0x01;
  private static final int RECV = 0x02;
  private static final int QUIT = 0x03;

  private static void connect() {
    System.out.println("Connecting...");
    usb = USB.waitForConnection(0, NXTConnection.RAW);
    out = usb.openDataOutputStream();
    in = usb.openDataInputStream();
  }

  private static void disconnect() throws java.io.IOException {
    System.out.println("Disconnecting...");
    out.close();
    in.close();
    USB.usbReset();
  }

  private static void initBot() {
    pilot = new DifferentialPilot(5.6, 11.2, Motor.C, Motor.B, false);
    pilot.setTravelSpeed(10);
    pilot.setRotateSpeed(40);
    sonar = new UltrasonicSensor(SensorPort.S4);
  }

  private static boolean check(int c) throws Exception {
    switch(c) {
      case QUIT:
        return true;
      case RECV:
        receive();
      break;
      case SEND:
        send();
      break;
    }
    return false;
  }

  private static void receive() throws Exception {
    int u = in.readInt();
    int d = in.readInt();
    pilot.travel(u*d, false);   
  }

  private static void send() throws Exception {
    int d = sonar.getDistance();
    out.writeInt(d);
    out.flush();
  }

  public static void main(String[] args) throws Exception {
    connect();
    initBot();
    System.out.println("Ready");
    while (true) {
      if (in.available() > 0) {
        int c = in.readInt();
        if (check(c)) {
          disconnect();
          break;
        }
      }
    }
  }
}
