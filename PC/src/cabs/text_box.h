#ifndef _CABS_WIDGETS_TEXT-BOX_H_
#define _CABS_WIDGETS_TEXT-BOX_H_

#include "widget.h"

class TextBox : public Widget
{
	private:
		std::string text;

	public:
		TextBox(int x, int y, const std::string& text);
		~TextBox();
};

#endif
