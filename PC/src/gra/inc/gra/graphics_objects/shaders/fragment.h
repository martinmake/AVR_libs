#ifndef _GRA_GRAPHICS_OBJECT_SHADER_FRAGMENT_H_
#define _GRA_GRAPHICS_OBJECT_SHADER_FRAGMENT_H_

#include "gra/graphics_objects/shaders/base.h"

namespace Gra
{
	namespace GraphicsObject
	{
		namespace Shader
		{
			class Fragment : public Shader::Base
			{
				public:
					Fragment(void);
					Fragment(const std::string& filepath_or_source);

					Fragment(const Fragment&  other);
					Fragment(      Fragment&& other);

					~Fragment(void);

				private:
					void copy(const Fragment&  other);
					void move(      Fragment&& other);
				public:
					Fragment& operator=(const Fragment&  rhs);
					Fragment& operator=(      Fragment&& rhs);
			};

			inline Fragment::Fragment(const Fragment&  other) : Fragment() { copy(          other ); }
			inline Fragment::Fragment(      Fragment&& other) : Fragment() { move(std::move(other)); }

			inline Fragment& Fragment::operator=(const Fragment&  rhs) { copy(          rhs ); return *this; }
			inline Fragment& Fragment::operator=(      Fragment&& rhs) { move(std::move(rhs)); return *this; }
		}
	}
}

#endif
