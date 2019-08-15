#ifndef _GRA_GRAPHICS_OBJECT_BASE_H_
#define _GRA_GRAPHICS_OBJECT_BASE_H_

#include <inttypes.h>

#include "gra/glstd.h"
#include "gra/gldebug.h"

namespace Gra
{
	namespace GraphicsObject
	{
		class Base
		{
			protected:
				unsigned int m_renderer_id;

			public:
				Base(void);

				Base(const Base&  other);
				Base(      Base&& other);

				virtual ~Base(void);

			public:
				virtual void   bind(void) const = 0;
				virtual void unbind(void) const = 0;

			public: // GETTERS
				unsigned int renderer_id(void) const;

			protected:
				void copy(const Base&  other);
				void move(      Base&& other);
		};

		inline unsigned int Base::renderer_id(void) const { return m_renderer_id; }

		inline Base::Base(const Base&  other) : Base() { copy(          other ); }
		inline Base::Base(      Base&& other) : Base() { move(std::move(other)); }
	}
}

#endif
