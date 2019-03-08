#ifndef _STANDARD_UTIL_H_
#define _STANDARD_UTIL_H_

#ifdef __cplusplus
extern "C" {
#endif

extern void wait_ms();
extern void wait_us();

#ifdef __cplusplus
}
#endif

#ifdef STANDARD_DEFAULT_WAIT
void wait_ms()
{
	_delay_ms(1);
}
void wait_us()
{
	_delay_us(1);
}
#endif

#endif
