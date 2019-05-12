#include <stdio.h>

#include "iusart.h"

void IUsart::sendf(size_t size, const char* format, ...)
{
	va_list ap;
	char* buf = new char[size];

	va_start(ap, format);
	vsnprintf(buf, size, format, ap);
	va_end(ap);

	*this << buf;

	delete[] buf;
}
