#ifndef _GPL_CORE_H_
#define _GPL_CORE_H_

#include <vector>
#include <string>
#include <memory>
#include <utility>
#include <assert.h>
#include <inttypes.h>

#include <sml/sml.h>

#include <gra/math.h>

namespace Gpl
{
	using Position = Gra::Math::vec2<unsigned int  >;
	using Size     = Gra::Math::vec2<unsigned int  >;
	using Color    = Gra::Math::vec4<         float>;
}

#endif
