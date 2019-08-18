#ifndef _ESL_EVENT_H_
#define _ESL_EVENT_H_

#include <type_traits>
#include <functional>

namespace Esl
{
	namespace Event
	{
		template <typename T>
		using Callback = std::function<bool (T&)>;

		template <typename TypeA, typename TypeB>
		bool is(const TypeB& e)
		{
			(void) e;
			return std::is_same<TypeA, TypeB>();
		}

		template <typename BaseType, typename ChildType>
		bool is_of(const ChildType& e)
		{
			(void) e;
			return std::is_base_of<BaseType, ChildType>();
		}

		template <typename BaseType, typename ChildType>
		bool is_in(const ChildType& e)
		{
			(void) e;
			return std::is_base_of<BaseType, ChildType>();
		}
	}
}

#endif
