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
				public: // CONSTRUCTORS
					Base(void);
					Base(GLenum initial_type);
					Base(GLenum initial_type, const std::string& filepath_or_source);

				public: // FUNCTIONS
					bool load(const std::string& filepath);

					void   bind(void) const override {}
					void unbind(void) const override {}

				public: // GETTERS
					const std::string& source(void) const;
				public: // SETTERS
					bool source(const std::string& new_source);

				protected:
					std::string m_source;
				private:
					GLenum m_type;

				DECLARATION_MANDATORY_INTERFACE(Base)
			};

			DEFINITION_MANDATORY(Base, other.m_type)
		}
	}
}

#endif
