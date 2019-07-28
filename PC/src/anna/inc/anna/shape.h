#ifndef _ANNA_SHAPE_H_
#define _ANNA_SHAPE_H_

#include <inttypes.h>
#include <iostream>

namespace Anna
{
	struct Shape
	{
		private:
			uint64_t m_width;
			uint64_t m_height;
			union
			{
				uint64_t m_depth;
				uint64_t m_channels_count;
			};
			uint64_t m_time;
		public:
			static const Shape INVALID;

		public:
			Shape(void);
			Shape(uint64_t initial_width, uint64_t initial_height = 1, uint64_t initial_depth = 1, uint64_t initial_time = 1);
			~Shape(void);

		public:
			bool is_valid(void) const;
			uint64_t hypervolume(void) const;

		public: // GETTERS
			uint64_t width         (void) const;
			uint64_t height        (void) const;
			uint64_t depth         (void) const;
			uint64_t channels_count(void) const;
			uint64_t time          (void) const;
		public: // SETTERS
			void width         (uint64_t new_width         );
			void height        (uint64_t new_height        );
			void depth         (uint64_t new_depth         );
			void channels_count(uint64_t new_channels_count);
			void time          (uint64_t new_time          );
	};

	inline bool Shape::is_valid(void) const { return m_width && m_height && m_depth && m_time; }
	inline uint64_t Shape::hypervolume(void) const { return m_width * m_height * m_depth * m_time; }

	// GETTERS
	inline uint64_t Shape::width         (void) const { return m_width;          }
	inline uint64_t Shape::height        (void) const { return m_height;         }
	inline uint64_t Shape::depth         (void) const { return m_depth;          }
	inline uint64_t Shape::channels_count(void) const { return m_channels_count; }
	inline uint64_t Shape::time          (void) const { return m_time;           }
	// SETTERS
	inline void Shape::width         (uint64_t new_width )         { m_width          = new_width;          }
	inline void Shape::height        (uint64_t new_height)         { m_height         = new_height;         }
	inline void Shape::depth         (uint64_t new_depth )         { m_depth          = new_depth;          }
	inline void Shape::channels_count(uint64_t new_channels_count) { m_channels_count = new_channels_count; }
	inline void Shape::time          (uint64_t new_time  )         { m_time           = new_time;           }
}

#endif
