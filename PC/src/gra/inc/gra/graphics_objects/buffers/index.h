#ifndef _GRA_GRAPHICS_OBJECT_BUFFER_INDEX_H_
#define _GRA_GRAPHICS_OBJECT_BUFFER_INDEX_H_

#include <inttypes.h>
#include <utility>
#include <vector>

#include "gra/gra.h"
#include "gra/graphics_objects/buffers/base.h"

namespace Gra
{
	namespace GraphicsObject
	{
		namespace Buffer
		{
			class Index : public Buffer::Base
			{
				public: // TYPES
					using type = uint32_t;

				public: // CONSTRUCTORS
					Index(void);
					Index(const std::vector<type>& initial_indices);

				public: // GETTERS
					const std::vector<type>& indices(void) const;
				public: // SETTERS
					void indices(const std::vector<type>& new_indices);

				private:
					std::vector<type> m_indices;

				DECLARATION_MANDATORY(Index)
			};

			// GETTERS
			DEFINITION_DEFAULT_GETTER(Index, indices, const std::vector<Index::type>&)

			DEFINITION_MANDATORY(Index, )
		}
	}
}

#endif
