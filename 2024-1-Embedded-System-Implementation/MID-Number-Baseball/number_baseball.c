/*
 * ATmega128A Number Baseball
 *
 * Input: four digit switches and an enter switch
 * Output: FND result and LED status
 * Flow: interrupt input -> duplicate check -> strike/ball comparison -> display
 */

/* Report page 11 */
⑤ strike&ball 둘다 발생: 4312 -> 2S2b 출력 + 빨간색 1개&노란색 1개 LED가 켜지는 것을 확인 가능⑥ OUT 발생: 5678 -> OU7 출력 + 빨간색 LED 2개 켜지는 것을 확인 가능⑦ 정답 입력: 4321 -> 정답값인 4321 출력 + 초록색 LED 1개가 켜지는 것을 확인 가능5. 코드 최종본#include <avr/io.h>#include <avr/interrupt.h>#define F_CPU 16000000UL#include <util/delay.h>#include <stdlib.h>unsigned char digit[10] = {0x3F, 0x06, 0x5B, 0x4F, 0x66, 0x6D, 0x7D, 0x07, 0x7F, 0x6F};unsigned char answer[4] = {1, 2, 3, 4};  //answer=4321unsigned char input[4] = {0, 0, 0, 0};unsigned char output[4] = {0, 0, 0, 0};unsigned char isInputReady = 0;int count = 0;/*확장*//*void initialize_answer() {for (int i = 0; i < 4; i++) {answer[i] = rand() % 10; // 0부터 9 사이의 랜덤 숫자 생성}}*/void display_fnd(){    for(int j = 0; j < 50; j++)    {    for(int i = 3; i >= 0; i--) {    PORTC = digit[input[i]];    PORTG = 1 << i;    _delay_ms(5);    }    }}

/* Report page 12 */
void display_fnd_out(){ if (output[3] == 4){ for(int j = 0; j < 50; j++)    //2초 설정{ for(int i = 3; i >= 0; i--) {PORTC = digit[answer[i]];PORTG = 1 << i;_delay_ms(5);}PORTA = 0x08;}}else if (output[3] == 0 && output[1] == 0){ for(int j = 0; j < 50; j++){ PORTC = 0x3F; PORTG = 1 << 3;_delay_ms(5);PORTC = 0x3e; PORTG = 1 << 2;_delay_ms(5);PORTC = 0x07; PORTG = 1 << 1;_delay_ms(5);PORTC = 0x00;PORTG = 1 << 0;_delay_ms(5);PORTA = 0x03;}}else{ for(int j = 0; j < 50; j++){

/* Report page 13 */
PORTC = digit[output[3]];PORTG = 1 << 3;_delay_ms(5);PORTC = 0x6D;PORTG = 1 << 2;_delay_ms(5);PORTC = digit[output[1]];PORTG = 1 << 1;_delay_ms(5);PORTC = 0x7C;PORTG = 1 << 0;_delay_ms(5);if (output[3] > 0 && output[1] > 0) {PORTA = 0x06; // S&B 모두 발생} else if (output[3] > 0) {PORTA = 0x02; // S만 발생} else if (output[1] > 0) {PORTA = 0x04; // B만 발생} else {PORTA = 0x00; // 둘다 발생X}}}}ISR(INT3_vect){ _delay_ms(100);if((PIND & 0x08) == 0x08)return;input[3]++;if (input[3] > 9)input[3] = 0;display_fnd();EIFR = EIFR | 1<<3;

/* Report page 14 */
}ISR(INT2_vect){ _delay_ms(100);if((PIND & 0x04) == 0x04)return;input[2]++;if (input[2] > 9)input[2] = 0;display_fnd();EIFR = EIFR | 1<<2;}ISR(INT1_vect){ _delay_ms(100);if((PIND & 0x02) == 0x02)return;input[1]++;if (input[1] > 9)input[1] = 0;display_fnd(); EIFR = EIFR | 1<<1;}ISR(INT0_vect){ _delay_ms(100);if((PIND & 0x01) == 0x01)return;input[0]++;if (input[0] > 9)input[0] = 0;

/* Report page 15 */
display_fnd(); EIFR = EIFR | 1<<0;}void check_result(){ int strike = 0, ball = 0;for (int i = 0; i < 4; i++){ if (input[i] == answer[i])strike++;else{ for (int j = 0; j < 4; j++){ if (input[i] == answer[j])ball++;}}}output[3] = strike;output[1] = ball;display_fnd_out(); }ISR(INT4_vect){ _delay_ms(100);if ((PINE & 0x10) == 0x10)return;for (int i = 0; i < 3; i++) {for (int j = i + 1; j < 4; j++) {if (input[i] == input[j]) {   //중복 발생 시 엔터 스위치 비활성화return;}}}isInputReady = 1;_delay_ms(1000);

/* Report page 16 */
EIFR |= (1 << INTF4); }int main(void){ DDRD = 0x00;DDRC = 0xFF;DDRG = 0xFF;DDRA = 0xff;EIMSK = 0x1F; // Enable INT0-INT4EICRA = 0x0A; // Falling Edge Triggersei(); // Enable global interrupts/*initialize_answer(); //확장*/display_fnd();while (1){ if (isInputReady){ check_result();isInputReady = 0;}}return 0;}
