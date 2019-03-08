#ifndef _YROBOT_CONNECTIONS_H_
#define _YROBOT_CONNECTIONS_H_

#include <avr/io.h>

#include <standard/standard.h>

/* # LEDs # */
// LED7
extern const BIT LED7;
// LED8
extern const BIT LED8;
/* # LEDs # */

/* # Buzzer # */
// AKU
extern const BIT BUZZER;
/* # Buzzer # */

/* # 7-Segment Displays # */
// LED6DIG1
extern const BIT LED6DIG1;
extern const BIT LED6DIG2;
// SSDBUS
extern const BIT SSDBUS_A;
extern const BIT SSDBUS_B;
extern const BIT SSDBUS_C;
extern const BIT SSDBUS_D;
extern const BIT SSDBUS_E;
extern const BIT SSDBUS_F;
extern const BIT SSDBUS_G;
extern const BIT SSDBUS_DP;
/* # 7-Segment Displays # */

/* # Touch switches & Trimmer # */
// SW2
extern const BIT SW2;
// SW3
extern const BIT SW3;
// TRIM1
extern const uint8_t TRIM1;
/* # Touch switches & Trimmer # */

/* # H-Bridge & Motors # */
// MOT1
extern const BIT MOTL_PWR;
extern const BIT MOTL_DIR;
extern const BIT MOTL_ENC;
// MOT2
extern const BIT MOTR_DIR;
extern const BIT MOTR_PWR;
extern const BIT MOTR_ENC;
/* # H-Bridge & Motors # */

#ifdef YROBOT_DEFAULT_CONNECTIONS
/* # LEDs # */
// LED7
const BIT LED7      = {&PORTA, PA5};
// LED8
const BIT LED8      = {&PORTA, PA6};
/* # LEDs # */

/* # Buzzer # */
// AKU
const BIT BUZZER    = {&PORTA, PA7};
/* # Buzzer # */

/* # 7-Segment Displays # */
// LED6DIG1
const BIT LED6DIG1  = {&PORTB, PB0};
const BIT LED6DIG2  = {&PORTB, PB1};
// SSDBUS
const BIT SSDBUS_A  = {&PORTC, PC0};
const BIT SSDBUS_B  = {&PORTC, PC1};
const BIT SSDBUS_C  = {&PORTC, PC2};
const BIT SSDBUS_D  = {&PORTC, PC3};
const BIT SSDBUS_E  = {&PORTC, PC4};
const BIT SSDBUS_F  = {&PORTC, PC5};
const BIT SSDBUS_G  = {&PORTC, PC6};
const BIT SSDBUS_DP = {&PORTC, PC7};
/* # 7-Segment Displays # */

/* # Touch switches & Trimmer # */
// SW2
const BIT SW2       = {&PORTB, PB2};
// SW3
const BIT SW3       = {&PORTB, PB3};
// TRIM1
const uint8_t TRIM1 = PA0;
/* # Touch switches & Trimmer # */

/* # H-Bridge & Motors # */
// MOT1
const BIT MOTL_PWR  = {&PORTD,  PD5};
const BIT MOTL_DIR  = {&PORTD,  PD6};
const BIT MOTL_ENC  = {&PORTD, INT0};
// MOT2
const BIT MOTR_PWR  = {&PORTD,  PD4};
const BIT MOTR_DIR  = {&PORTD,  PD7};
const BIT MOTR_ENC  = {&PORTD, INT1};
/* # H-Bridge & Motors # */
#endif

#endif
