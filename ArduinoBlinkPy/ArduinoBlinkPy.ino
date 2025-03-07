
void setup() {
    Serial.begin(9600);
    pinMode(LED_BUILTIN, OUTPUT);
}

void loop() {
    if (Serial.available() > 0) {
        char command = Serial.read();
        if (command == '1') {
            digitalWrite(LED_BUILTIN, HIGH); // Turn LED on
        } else if (command == '0') {
            digitalWrite(LED_BUILTIN, LOW); // Turn LED off
        }
    }
}
