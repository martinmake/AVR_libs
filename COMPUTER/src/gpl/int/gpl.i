%module gpl

%include <std_vector.i>
%include <std_string.i>

%include <gra/gra.h>
%include <gra/math.h>
%include <gra/vertex_buffer.h>
%include <gra/vertex_buffer_layout.h>
%include <gra/vertex_array.h>
%include <gra/index_buffer.h>
%include <gra/shader.h>
%include <gra/texture.h>
%include <gra/renderer.h>

%template(vec3) Gra::Math::vec3<float>;
%template(vec4) Gra::Math::vec4<float>;

%{
#include <gra/gra.h>
#include <gra/math.h>
#include <gra/vertex_buffer.h>
#include <gra/vertex_buffer_layout.h>
#include <gra/vertex_array.h>
#include <gra/index_buffer.h>
#include <gra/shader.h>
#include <gra/texture.h>
#include <gra/renderer.h>

#include "gpl/gpl.h"
#include "gpl/primitive.h"
#include "gpl/primitives/point.h"
#include "gpl/canvas.h"
%}

%include "gpl/gpl.h"
%include "gpl/primitive.h"
%include "gpl/primitives/base.h"
%include "gpl/primitives/point.h"
%include "gpl/canvas.h"
