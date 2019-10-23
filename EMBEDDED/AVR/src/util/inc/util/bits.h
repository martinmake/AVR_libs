#ifndef _UTIL_BITS_H_
#define _UTIL_BITS_H_

#define BIT(index) (1 << index)

#define SET(   port, index) port |=  BIT(index)
#define CLEAR( port, index) port &= ~BIT(index)
#define TOOGLE(port, index) port ^=  BIT(index)

#define IS_SET(  port, index) (  port & BIT(index) )
#define IS_CLEAR(port, index) (!(port & BIT(index)))

#define BIN =  2,
#define OCT =  8,
#define DEC = 10,
#define HEX = 16,

#define ON  true
#define OFF false

#define ENABLE  true
#define DISABLE false

#define OK  true
#define ERR false

#define HIGH true
#define LOW  false

#define ACK  true
#define NACK false

#endif
