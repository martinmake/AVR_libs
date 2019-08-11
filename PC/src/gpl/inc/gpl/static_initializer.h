#ifndef _GPL_STATIC_INITIALIZER_H_
#define _GPL_STATIC_INITIALIZER_H_

namespace Gpl
{
	class StaticInitializer
	{
		public:
			StaticInitializer(void);
			~StaticInitializer(void);

		private:
			void initialize(void);
	};
}

#endif
