#ifndef _LOGGER_CONSOLE_H_
#define _LOGGER_CONSOLE_H_

#include "logger/loggers/base.h"

namespace Logger
{
	class Console : public Base
	{
		public:
			Console(void);
			Console(const std::string& initial_name);

			Console(const Console&  other);
			Console(      Console&& other);

			~Console(void);

		protected:
			void copy(const Console&  other);
			void move(      Console&& other);
		public:
			Console& operator=(const Console&  rhs);
			Console& operator=(      Console&& rhs);
	};

	inline Console::Console(const Console&  other) : Base() { copy(          other ); }
	inline Console::Console(      Console&& other) : Base() { move(std::move(other)); }

	inline Console& Console::operator=(const Console&  rhs) { copy(          rhs ); return *this; }
	inline Console& Console::operator=(      Console&& rhs) { move(std::move(rhs)); return *this; }
}

#endif
