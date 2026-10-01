/* ATmega128A Number Baseball
 * 입력: 4자리 스위치와 엔터 스위치
 * 출력: FND 결과 표시와 LED 상태 표시
 * 흐름: 인터럽트 입력 -> 중복 검사 -> Strike/Ball 계산 -> 출력
 */

#include <avr/io.h>
#include <avr/interrupt.h>
#define F_CPU 16000000UL
#include <util/delay.h>
#include <stdlib.h>
unsigned char digit[10] = {0x3F, 0x06, 0x5B, 0x4F, 0x66, 0x6D, 0x7D, 0x07, 0x7F, 0x6F};
unsigned char answer[4] = {1, 2, 3, 4}; //answer=4321
unsigned char input[4] = {0, 0, 0, 0};
unsigned char output[4] = {0, 0, 0, 0};
unsigned char isInputReady = 0;
int count = 0;
/*확장*/
/*
void initialize_answer() {
for (int i = 0; i < 4; i++) {
answer[i] = rand() % 10; // 0부터 9 사이의 랜덤 숫자 생성
}
}
*/
void display_fnd()
{
for(int j = 0; j < 50; j++)
{
for(int i = 3; i >= 0; i--) {
PORTC = digit[input[i]];
PORTG = 1 << i;
_delay_ms(5);
}
}
}

/* 정답, OUT, Strike/Ball 결과 표시 */
void display_fnd_out()
{
if (output[3] == 4)
{
for(int j = 0; j < 50; j++) //2초 설정
{
for(int i = 3; i >= 0; i--) {
PORTC = digit[answer[i]];
PORTG = 1 << i;
_delay_ms(5);
}
PORTA = 0x08;
}
}
else if (output[3] == 0 && output[1] == 0)
{
for(int j = 0; j < 50; j++)
{
PORTC = 0x3F;
PORTG = 1 << 3;
_delay_ms(5);
PORTC = 0x3e;
PORTG = 1 << 2;
_delay_ms(5);
PORTC = 0x07;
PORTG = 1 << 1;
_delay_ms(5);
PORTC = 0x00;
PORTG = 1 << 0;
_delay_ms(5);
PORTA = 0x03;
}
}
else
{
for(int j = 0; j < 50; j++)
{

/* 4번째 자리 입력 인터럽트 */
PORTC = digit[output[3]];
PORTG = 1 << 3;
_delay_ms(5);
PORTC = 0x6D;
PORTG = 1 << 2;
_delay_ms(5);
PORTC = digit[output[1]];
PORTG = 1 << 1;
_delay_ms(5);
PORTC = 0x7C;
PORTG = 1 << 0;
_delay_ms(5);
if (output[3] > 0 && output[1] > 0) {
PORTA = 0x06; // S&B 모두 발생
} else if (output[3] > 0) {
PORTA = 0x02; // S만 발생
} else if (output[1] > 0) {
PORTA = 0x04; // B만 발생
} else {
PORTA = 0x00; // 둘다 발생X
}
}
}
}
ISR(INT3_vect)
{
_delay_ms(100);
if((PIND & 0x08) == 0x08)
return;
input[3]++;
if (input[3] > 9)
input[3] = 0;
display_fnd();
EIFR = EIFR | 1<<3;

/* 각 자리 입력 인터럽트 */
}
ISR(INT2_vect)
{
_delay_ms(100);
if((PIND & 0x04) == 0x04)
return;
input[2]++;
if (input[2] > 9)
input[2] = 0;
display_fnd();
EIFR = EIFR | 1<<2;
}
ISR(INT1_vect)
{
_delay_ms(100);
if((PIND & 0x02) == 0x02)
return;
input[1]++;
if (input[1] > 9)
input[1] = 0;
display_fnd();
EIFR = EIFR | 1<<1;
}
ISR(INT0_vect)
{
_delay_ms(100);
if((PIND & 0x01) == 0x01)
return;
input[0]++;
if (input[0] > 9)
input[0] = 0;

/* Strike/Ball 계산과 엔터 입력 확정 */
display_fnd();
EIFR = EIFR | 1<<0;
}
void check_result()
{
int strike = 0, ball = 0;
for (int i = 0; i < 4; i++)
{
if (input[i] == answer[i])
strike++;
else
{
for (int j = 0; j < 4; j++)
{
if (input[i] == answer[j])
ball++;
}
}
}
output[3] = strike;
output[1] = ball;
display_fnd_out();
}
ISR(INT4_vect)
{
_delay_ms(100);
if ((PINE & 0x10) == 0x10)
return;
for (int i = 0; i < 3; i++) {
for (int j = i + 1; j < 4; j++) {
if (input[i] == input[j]) { //중복 발생 시 엔터 스위치 비활성화
return;
}
}
}
isInputReady = 1;
_delay_ms(1000);

/* INT0~INT4 초기화와 메인 루프 */
EIFR |= (1 << INTF4);
}
int main(void)
{
DDRD = 0x00;
DDRC = 0xFF;
DDRG = 0xFF;
DDRA = 0xff;
EIMSK = 0x1F; // Enable INT0-INT4
EICRA = 0x0A; // Falling Edge Trigger
sei(); // Enable global interrupts
/*
initialize_answer(); //확장
*/
display_fnd();
while (1)
{
if (isInputReady)
{
check_result();
isInputReady = 0;
}
}
return 0;
}