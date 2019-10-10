#ifndef _LOGGER_BASE_H_
#define _LOGGER_BASE_H_

#include <inttypes.h>
#include <stdarg.h>
#include <memory>
#include <utility>

#include <spdlog/logger.h>

namespace Logger
{
	class Base
	{
		protected:
			std::shared_ptr<spdlog::logger> m_underlying_logger;
			std::string                     m_name;

		public:
			Base(void);
			Base(const std::string& initial_name);

			Base(const Base&  other);
			Base(      Base&& other);

			virtual ~Base(void);

		protected:
			void init(void);

		public:
			const std::shared_ptr<spdlog::logger>& underlying_logger(void) const;

		protected:
			void copy(const Base&  other);
			void move(      Base&& other);
	};

	inline const std::shared_ptr<spdlog::logger>& Base::underlying_logger(void) const { return m_underlying_logger; }

	inline Base::Base(const Base&  other) : Base() { copy(          other ); }
	inline Base::Base(      Base&& other) : Base() { move(std::move(other)); }
}

#endif
