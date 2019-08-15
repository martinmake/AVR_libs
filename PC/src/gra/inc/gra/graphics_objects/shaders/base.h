#ifndef _GRA_GRAPHICS_OBJECT_SHADER_BASE_H_
#define _GRA_GRAPHICS_OBJECT_SHADER_BASE_H_

#include <sstream>
#include <fstream>

#include "gra/glstd.h"
#include "gra/gldebug.h"

#include "gra/graphics_objects/base.h"

namespace Gra
{
	namespace GraphicsObject
	{
		namespace Shader
		{
			class Base : public GraphicsObject::Base
			{
				private:
					GLenum m_type;
				protected:
					std::string m_source;

				public:
					Base(void);
					Base(GLenum initial_type);
					Base(GLenum initial_type, const std::string& filepath_or_source);

					Base(const Base&  other);
					Base(      Base&& other);

					~Base(void);

				public:
					bool load(const std::string& filepath);

				public:
					void   bind(void) const override {}
					void unbind(void) const override {}

				public: // GETTERS
					const std::string& source(void) const;
				public: // SETTERS
					bool source(const std::string& new_source);


				protected:
					void copy(const Base&  other);
					void move(      Base&& other);
				public:
					Base& operator=(const Base&  rhs);
					Base& operator=(      Base&& rhs);
			};

			inline Base::Base(const Base&  other) : Base(other.m_type) { copy(          other ); }
			inline Base::Base(      Base&& other) : Base(other.m_type) { move(std::move(other)); }

			inline Base& Base::operator=(const Base&  rhs) { copy(          rhs ); return *this; }
			inline Base& Base::operator=(      Base&& rhs) { move(std::move(rhs)); return *this; }
		}
	}
}

#endif
