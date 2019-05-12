#ifndef _YROBOT_CONNECTIONS_H_
#define _YROBOT_CONNECTIONS_H_

#include <avr/io.h>

#include <standard/standard.h>

/* # LEDs # */
// LED7
extern const Pin LED7;
// LED8
extern const Pin LED8;
/* # LEDs # */

/* # Buzzer # */
// AKU
extern const Pin BUZZER;
/* # Buzzer # */

/* # 7-Segment Displays # */
// LED6DIG1
extern const Pin LED6DIG1;
extern const Pin LED6DIG2;
// SSDBUS
extern const Pin SSDBUS_A;
extern const Pin SSDBUS_B;
extern const Pin SSDBUS_C;
extern const Pin SSDBUS_D;
extern const Pin SSDBUS_E;
extern const Pin SSDBUS_F;
extern const Pin SSDBUS_G;
extern const Pin SSDBUS_DP;
/* # 7-Segment Displays # */

/* # Touch switches & Trimmer # */
// SW2
extern const Pin SW2;
// SW3
extern const Pin SW3;
// TRIM1
extern const Pin TRIM1;
/* # Touch switches & Trimmer # */

/* # H-Bridge & Motors # */
// MOT1
extern const Pin MOTL_PWR;
extern const Pin MOTL_DIR;
extern const Pin MOTL_ENC;
// MOT2
extern const Pin MOTR_PWR;
extern const Pin MOTR_DIR;
extern const Pin MOTR_ENC;
/* # H-Bridge & Motors # */

#ifdef YROBOT_DEFAULT_CONNECTIONS
/* # LEDs # */
// LED7
const Pin LED7      = {&PORTA, PA5};
// LED8
const Pin LED8      = {&PORTA, PA6};
/* # LEDs # */

/* # Buzzer # */
// AKU
const Pin BUZZER    = {&PORTA, PA7};
/* # Buzzer # */

/* # 7-Segment Displays # */
// LED6DIG1
const Pin LED6DIG1  = {&PORTB, PB0};
const Pin LED6DIG2  = {&PORTB, PB1};
// SSDBUS
const Pin SSDBUS_A  = {&PORTC, PC0};
const Pin SSDBUS_B  = {&PORTC, PC1};
const Pin SSDBUS_C  = {&PORTC, PC2};
const Pin SSDBUS_D  = {&PORTC, PC3};
const Pin SSDBUS_E  = {&PORTC, PC4};
const Pin SSDBUS_F  = {&PORTC, PC5};
const Pin SSDBUS_G  = {&PORTC, PC6};
const Pin SSDBUS_DP = {&PORTC, PC7};
/* # 7-Segment Displays # */

/* # Touch switches & Trimmer # */
// SW2
const Pin SW2       = {&PORTB, PB2};
// SW3
const Pin SW3       = {&PORTB, PB3};
// TRIM1
const Pin TRIM1     = {&PORTA, PA0};
/* # Touch switches & Trimmer # */

/* # H-Bridge & Motors # */
// MOT1
const Pin MOTL_PWR  = {&PORTD,  PD5};
const Pin MOTL_DIR  = {&PORTD,  PD6};
const Pin MOTL_ENC  = {&PORTD, INT0};
// MOT2
const Pin MOTR_PWR  = {&PORTD,  PD4};
const Pin MOTR_DIR  = {&PORTD,  PD7};
const Pin MOTR_ENC  = {&PORTD, INT1};
/* # H-Bridge & Motors # */
#endif

#endif
