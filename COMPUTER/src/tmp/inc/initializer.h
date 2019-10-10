#ifndef _INITIALIZER_H_
#define _INITIALIZER_H_

#include <iostream>

namespace Tmp
{
	struct Initializer
	{
		 Initializer(void) { std::cout << "INIT"   << std::endl; }
		~Initializer(void) { std::cout << "DEINIT" << std::endl; }
	};
}

#endif
