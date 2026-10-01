/* ATmega128A Mole Game
 * 입력: 게임 스위치, 초음파 센서, 사운드 센서
 * 출력: LED, FND, CLCD, 부저
 * 흐름: 랜덤 타깃 -> 인터럽트 입력 -> 점수 계산 -> 시간/피버타임 처리
 * 여러 주변 장치의 ISR 충돌을 줄이기 위해 기능별 인터럽트를 분리함.
 */

/* LED 세트 1: 외부 인터럽트와 점수 카운트 */
button_pressed_1 = 1;
if (led_state_1 == (1 << RED_LED_1)) {
red_count++; // 빨간색 LED일 때 카운트 증가
} else if (led_state_1 == (1 << YELLOW_LED_1)) {
yellow_count++; // 노란색 LED일 때 카운트 증가
} else if (led_state_1 == (1 << GREEN_LED_1)) {
green_count++; // 초록색 LED일 때 카운트 증가
} else {
error_count++; // LED가 아무것도 켜져 있지 않은 경우 에러 카운
트 증가
}
}
EIFR |= (1 << INTF0); // 인터럽트 플래그 클리어
}
void setup_1() {
DDRB |= (1 << RED_LED_1) | (1 << YELLOW_LED_1) | (1 << GREEN_LED_1);
DDRD &= ~(1 << SWITCH_1);
PORTD |= (1 << SWITCH_1);
EICRA |= (1 << ISC01);
EIMSK |= (1 << INT0);
sei();
srand(0);
}
void loop_1() {
uint8_t led_choice_1 = rand() % 10;
if (led_choice_1 < 3) {
led_state_1 = (1 << RED_LED_1);
} else if (led_choice_1 < 9) {
led_state_1 = (1 << YELLOW_LED_1);
} else {
led_state_1 = (1 << GREEN_LED_1);
}
PORTB |= led_state_1;
for (uint16_t i = 0; i < 100; i++) {
_delay_ms(10);
if (button_pressed_1) {
button_pressed_1 = 0;

/* LED 세트 2 처리 */
PORTB &= ~led_state_1;
return;
}
}
PORTB &= ~led_state_1;
}
#define RED_LED_2 PB3
#define YELLOW_LED_2 PB4
#define GREEN_LED_2 PB5
#define SWITCH_2 PD1
volatile uint8_t led_state_2 = 0;
volatile uint8_t button_pressed_2 = 0;
ISR(INT1_vect) {
_delay_ms(50); // 디바운싱을 위한 짧은 지연
if ((PIND & (1 << SWITCH_2)) == 0) {
button_pressed_2 = 1;
if (led_state_2 == (1 << RED_LED_2)) {
red_count++; // 빨간색 LED일 때 카운트 증가
} else if (led_state_2 == (1 << YELLOW_LED_2)) {
yellow_count++; // 노란색 LED일 때 카운트 증가
} else if (led_state_2 == (1 << GREEN_LED_2)) {
green_count++; // 초록색 LED일 때 카운트 증가
} else {
error_count++; // LED가 아무것도 켜져 있지 않은 경우 에러 카운
트 증가
}
}
EIFR |= (1 << INTF1); // 인터럽트 플래그 클리어
}
void setup_2() {
DDRB |= (1 << RED_LED_2) | (1 << YELLOW_LED_2) | (1 << GREEN_LED_2);
DDRD &= ~(1 << SWITCH_2);
PORTD |= (1 << SWITCH_2);
EICRA |= (1 << ISC11);
EIMSK |= (1 << INT1);
sei();

/* LED 세트별 랜덤 점등 */
srand(1);
}
void loop_2() {
uint8_t led_choice_2 = rand() % 10;
if (led_choice_2 < 3) {
led_state_2 = (1 << RED_LED_2);
} else if (led_choice_2 < 9) {
led_state_2 = (1 << YELLOW_LED_2);
} else {
led_state_2 = (1 << GREEN_LED_2);
}
PORTB |= led_state_2;
for (uint16_t i = 0; i < 100; i++) {
_delay_ms(10);
if (button_pressed_2) {
button_pressed_2 = 0;
PORTB &= ~led_state_2;
return;
}
}
PORTB &= ~led_state_2;
}
#define RED_LED_3 PD5
#define YELLOW_LED_3 PD6
#define GREEN_LED_3 PD7
#define SWITCH_3 PD2
volatile uint8_t led_state_3 = 0;
volatile uint8_t button_pressed_3 = 0;
ISR(INT2_vect) {
_delay_ms(50); // 디바운싱을 위한 짧은 지연
if ((PIND & (1 << SWITCH_3)) == 0) {
button_pressed_3 = 1;
if (led_state_3 == (1 << RED_LED_3)) {
red_count++; // 빨간색 LED일 때 카운트 증가
} else if (led_state_3 == (1 << YELLOW_LED_3)) {
yellow_count++; // 노란색 LED일 때 카운트 증가

/* 세 번째 LED 세트와 스위치 입력 */
} else if (led_state_3 == (1 << GREEN_LED_3)) {
green_count++; // 초록색 LED일 때 카운트 증가
} else {
error_count++; // LED가 아무것도 켜져 있지 않은 경우 에러 카운
트 증가
}
}
EIFR |= (1 << INTF2); // 인터럽트 플래그 클리어
}
void setup_3() {
DDRD |= (1 << RED_LED_3) | (1 << YELLOW_LED_3) | (1 << GREEN_LED_3);
DDRD &= ~(1 << SWITCH_3);
PORTD |= (1 << SWITCH_3);
EICRA |= (1 << ISC21);
EIMSK |= (1 << INT2);
sei();
srand(2);
}
void loop_3() {
uint8_t led_choice_3 = rand() % 10;
if (led_choice_3 < 3) {
led_state_3 = (1 << RED_LED_3);
} else if (led_choice_3 < 9) {
led_state_3 = (1 << YELLOW_LED_3);
} else {
led_state_3 = (1 << GREEN_LED_3);
}
PORTD |= led_state_3;
for (uint16_t i = 0; i < 100; i++) {
_delay_ms(10);
if (button_pressed_3) {
button_pressed_3 = 0;
PORTD &= ~led_state_3;
return;
}
}

/* 점수 저장과 최고 점수 관리 */
PORTD &= ~led_state_3;
}
volatile int score = 0;
volatile int score_list_1[10]; // 최대 10번 플레이까지 저장할 수 있도록 설정
volatile int score_list_2[10];
volatile int score_count_1 = 0; // 플레이 횟수 카운트
volatile int score_count_2 = 0;
volatile int best_score_1 = 0; // 최고 점수 저장 변수
volatile int best_score_2 = 0;
void update_score() {
score = -10 * (red_count + error_count) + 10 * yellow_count + 50 *
green_count;
}
unsigned char digit[10] = {0x3F, 0x06, 0x5B, 0x4F, 0x66, 0x6D, 0x7D, 0x07, 0x7F, 0x6F};
void display_score(int score) {
unsigned char fnd_output[4];
if (score < 0) {
fnd_output[3] = 0x40; // '-' 출력
score = -score; // 절대값으로 변경
} else {
fnd_output[3] = digit[score / 1000]; // 천의 자리
}
fnd_output[2] = digit[(score / 100) % 10]; // 백의 자리
fnd_output[1] = digit[(score / 10) % 10]; // 십의 자리
fnd_output[0] = digit[score % 10]; // 일의 자리
for (int j = 0; j < 50; j++) {
for (int i = 3; i >= 0; i--) {
PORTC = fnd_output[i];
PORTG = 1 << i;
_delay_ms(5);
}
}
}
void record_score_1(int score) {

/* 점수 순위 계산 */
if (score_count_1 < 10) {
score_list_1[score_count_1] = score;
score_count_1++;
}
if (score > best_score_1) {
best_score_1 = score;
}
}
void record_score_2(int score) {
if (score_count_2 < 10) {
score_list_2[score_count_2] = score;
score_count_2++;
}
if (score > best_score_2) {
best_score_2 = score;
}
}
int calculate_rank_1(int score, volatile int *score_list, int score_count) {
int rank_1 = 1;
for (int i = 0; i < score_count; ++i) {
if (score > score_list[i]) {
rank_1++;
}
}
return rank_1;
}
int calculate_rank_2(int score, volatile int *score_list, int score_count) {
int rank_2 = 1;
for (int i = 0; i < score_count; ++i) {
if (score > score_list[i]) {
rank_2++;
}
}
return rank_2;
}
void init_uart1();
void putchar1(char c);
char getchar1();

/* UART 통신 설정 */
char available1();
void init_uart1()
{
UCSR1B = 0x18;
UCSR1C = 0x06;
UBRR1H = 0;
UBRR1L = 8;
}
void putchar1(char c) {
while (!(UCSR1A & (1<<UDRE1)))
;
UDR1 = c;
}
char getchar1()
{
return(UDR1); // 수신된 데이터를 가져옴
}
char available() // 문자 입력을 확인
{
if (UCSR1A & (1<<RXC1))
return(1); // 입력된 문자가 있으면 1 리턴
else
return(0); // 입력된 문자가 없으면 0 리턴
}
#define S_LEVEL_3 573
#define S_LEVEL_2 328
#define S_LEVEL_1 82
void init_adc();
unsigned short read_adc();
void init_adc()
{
ADMUX = 0x41;
ADCSRA = 0x87;
}
unsigned short read_adc()

/* ADC 기반 사운드 센서 점수 */
{
unsigned char adc_low, adc_high;
unsigned short value;
ADCSRA |= 0x40;
while((ADCSRA & 0x10) != 0x10)
{
;
}
adc_low = ADCL;
adc_high = ADCH;
value = (adc_high<<8) | adc_low;
return value;
}
void sound_score_f(unsigned short value)
{
if (value >= S_LEVEL_3) sound_score += 30;
else if (value >= S_LEVEL_2) sound_score += 20;
else if (value >= S_LEVEL_1) sound_score += 10;
}
#define BIT4_LINE2_DOT58 0x28
#define DISPON_CUROFF_BLKOFF 0x0C
#define DISPOFF_CUROFF_BLKOFF 0x08 // display off, cursor off, blink off
#define INC_NOSHIFT 0x06
#define DISPCLEAR 0x01
#define CUR1LINE 0x80 // 커서를 첫번째 줄 처음으로 위치
#define CUR2LINE 0xC0 // 커서를 두번째 줄 처음으로 위치
void CLCD_cmd(char);
void CLCD_data(char);
void CLCD_puts(char *);
void CLCD_cmd(char cmd)
{
PORTF = 0x00; // RS = 0, RW = 0, EN = 0
_delay_us(1);
PORTF = 0x10; // EN = 1
PORTE = cmd & 0xf0; // 상위 4비트 전송
PORTF = 0x00; // EN = 0
_delay_us(2);
PORTF = 0x10; // EN = 1
PORTE = (cmd << 4) & 0xf0; // 하위 4비트 전송
PORTF = 0x00; // EN = 0

/* CLCD 출력 함수 */
_delay_us(2);
_delay_ms(1);
}
void CLCD_data(char data)
{
PORTF = 0x04; // RS = 1, RW = 0, EN = 0
_delay_us(1);
PORTF = 0x14; // EN = 1
PORTE = data & 0xf0; // 상위 4비트 전송
PORTF = 0x04; // EN = 0
_delay_us(2);
PORTF = 0x14; // EN = 1
PORTE = (data << 4) & 0xf0; // 하위 4비트 전송
PORTF = 0x04; // EN = 0
_delay_us(2);
_delay_ms(1);
}
void CLCD_puts(char *ptr)
{
while(*ptr != NULL)
{
CLCD_data(*ptr++);
}
}
void display_rank_best(int level, int rank, int best_score) {
char buf[32]; // 문자열 버퍼 크기 조정
// 순위 표시
if (level == 1) {
sprintf(buf, "RANK 1: %d/%d ", rank, score_count_1);
} else if (level == 2) {
sprintf(buf, "RANK 2: %d/%d ", rank, score_count_2);
}
CLCD_cmd(CUR1LINE);
CLCD_puts(buf);
// 최고 점수 표시
if (level == 1) {
sprintf(buf, "BEST 1: %d ", best_score);
} else if (level == 2) {

/* 레벨 선택과 메인 흐름 */
sprintf(buf, "BEST 2: %d ", best_score);
}
CLCD_cmd(CUR2LINE);
CLCD_puts(buf);
}
char sec1[] = "30s : select 1 ";
char sec2[] = "1m : select 2 ";
int main() {
// UART 초기화
init_uart1();
char c;
// 기타 필요한 설정들
setup_1();
setup_2();
setup_3();
// FND와 관련된 포트 설정
DDRC = 0xFF; // PORTC 출력
DDRG = 0xFF; // PORTG 출력
DDRA = 0xff;
DDRE = 0xff;
DDRF = 0b00011100;
unsigned short value;
init_adc();
CLCD_cmd(DISPCLEAR);
CLCD_cmd(CUR1LINE); // 커서를 첫번째 줄 처음으로 위치
CLCD_puts(sec1); // 첫째 줄 출력
CLCD_cmd(CUR2LINE); // 커서를 두번째 줄 처음으로 위치
CLCD_puts(sec2);
while (1) {
if (available()) {
c = getchar1();
putchar1(c);
red_count = 0;

/* 게임 진행과 제한 시간 */
yellow_count = 0;
green_count = 0;
error_count = 0;
sound_score = 0;
if (c == '1') {
for (int i = 0; i < 5; ++i) {
// LED를 켬
PORTA |= (1 << PA0);
_delay_ms(500);
// LED를 끔
PORTA &= ~(1 << PA0);
_delay_ms(500);
}
// 30초 동안 코드 실행
for (uint16_t i = 0; i < 12; ++i) {
uint8_t set_choice = rand() % 3;
switch (set_choice) {
case 0:
loop_1();
break;
case 1:
loop_2();
break;
case 2:
loop_3();
break;
}
update_score();
display_score(score);
_delay_ms(100);
}
for (uint16_t i = 0; i < 3; ++i) {
PORTA |= (1 << PA2);
uint8_t set_choice = rand() % 3;
switch (set_choice) {
case 0:
loop_1();
break;
case 1:
loop_2();

/* 게임 종료 처리 */
break;
case 2:
loop_3();
break;
}
update_score();
display_score(score);
_delay_ms(100);
}
PORTA &= ~(1 << PA2);
for (int i = 0; i < 5; ++i) {
// LED를 켬
PORTA |= (1 << PA0);
_delay_ms(500);
// LED를 끔
PORTA &= ~(1 << PA0);
_delay_ms(500);
}
for (int seconds = 0; seconds < 10; seconds++) {
value = read_adc();
sound_score_f(value);
display_score(sound_score);
_delay_ms(50);
}
record_score_1(score + sound_score);
for (int i = 0; i < 5; ++i)
display_score(score+sound_score);
display_rank_best(1, calculate_rank_1(score +
sound_score, score_list_1, score_count_1), best_score_1);
}
else if (c == '2') {
for (int i = 0; i < 5; ++i) {
// LED를 켬
PORTA |= (1 << PA1);
_delay_ms(500);
// LED를 끔
PORTA &= ~(1 << PA1);
_delay_ms(500);
}
// 1분 동안 코드 실행

/* 레벨 2 게임 진행 */
for (uint16_t i = 0; i < 27; ++i) {
uint8_t set_choice = rand() % 3;
switch (set_choice) {
case 0:
loop_1();
break;
case 1:
loop_2();
break;
case 2:
loop_3();
break;
}
update_score();
display_score(score);
_delay_ms(100);
}
for (uint16_t i = 0; i < 3; ++i) {
PORTA |= (1 << PA2);
uint8_t set_choice = rand() % 3;
switch (set_choice) {
case 0:
loop_1();
break;
case 1:
loop_2();
break;
case 2:
loop_3();
break;
}
update_score();
display_score(score);
_delay_ms(100);
}
PORTA &= ~(1 << PA2);
for (int i = 0; i < 5; ++i) {
// LED를 켬
PORTA |= (1 << PA1);
_delay_ms(500);
// LED를 끔
PORTA &= ~(1 << PA1);

/* 피버타임 점수와 최종 출력 */
_delay_ms(500);
}
for (int seconds = 0; seconds < 10; seconds++) {
value = read_adc();
sound_score_f(value);
display_score(sound_score);
_delay_ms(50);
}
record_score_2(score + sound_score);
for (int i = 0; i < 5; ++i)
display_score(score+sound_score);
display_rank_best(2, calculate_rank_2(score +
sound_score, score_list_2, score_count_2), best_score_2);
}
_delay_ms(10000);
CLCD_cmd(DISPCLEAR);
CLCD_cmd(CUR1LINE); // 커서를 첫번째 줄 처음으로 위치
CLCD_puts(sec1); // 첫째 줄 출력
CLCD_cmd(CUR2LINE); // 커서를 두번째 줄 처음으로 위치
CLCD_puts(sec2);
}
}
return 0;
}
