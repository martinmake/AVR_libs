#include "logging.h"

#include "gra/renderer.h"

namespace Gra
{
	Renderer::Renderer(void)
	{
		TRACE("RENDERER: CONSTRUCTED: {0}", (void*) this);
	}

	Renderer::~Renderer(void)
	{
		TRACE("RENDERER: DESTRUCTED: {0}", (void*) this);
	}
}
