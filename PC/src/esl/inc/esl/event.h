#ifndef _ESL_EVENT_H_
#define _ESL_EVENT_H_

#include <type_traits>

namespace Esl
{
	namespace Event
	{
		template <typename BaseType, typename ChildType>
		bool is_of(const ChildType& e)
		{
			(void) e;
			return std::is_base_of<BaseType, ChildType>();
		}
	}
}

#endif
