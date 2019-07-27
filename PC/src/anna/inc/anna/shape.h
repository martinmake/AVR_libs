#ifndef _ANNA_SHAPE_H_
#define _ANNA_SHAPE_H_

#include <inttypes.h>
#include <iostream>

namespace Anna
{
	struct Shape
	{
		private:
			uint16_t m_width;
			uint16_t m_height;
			uint16_t m_channel_count;
			uint16_t m_time;
		public:
			static const Shape INVALID;

		public:
			Shape(void);
			Shape(uint16_t initial_width, uint16_t initial_height = 1, uint16_t initial_channel_count = 1, uint16_t initial_time = 1);
			~Shape(void);

		public:
			bool is_valid(void) const;

		public: // GETTERS
			uint16_t width         (void) const;
			uint16_t height        (void) const;
			uint16_t channel_count (void) const;
			uint16_t time          (void) const;
		public: // SETTERS
			void width         (uint16_t new_width         );
			void height        (uint16_t new_height        );
			void channel_count (uint16_t new_channel_count );
			void time          (uint16_t new_time          );

	};

	inline bool Shape::is_valid(void) const { return m_width && m_height && m_channel_count && m_time; }

	// GETTERS
	inline uint16_t Shape::width         (void) const { return m_width;          }
	inline uint16_t Shape::height        (void) const { return m_height;         }
	inline uint16_t Shape::channel_count (void) const { return m_channel_count;  }
	inline uint16_t Shape::time          (void) const { return m_time;           }

	// SETTERS
	inline void Shape::width         (uint16_t new_width         ) { m_width          = new_width;          }
	inline void Shape::height        (uint16_t new_height        ) { m_height         = new_height;         }
	inline void Shape::channel_count (uint16_t new_channel_count ) { m_channel_count  = new_channel_count;  }
	inline void Shape::time          (uint16_t new_time          ) { m_time           = new_time;           }
}

#endif
