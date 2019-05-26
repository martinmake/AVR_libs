{
using namespace Cabs::Positions;
using namespace Cabs::Colors;

widget1.position({LEFT, TOP});
widget1.size({21, 20});
widget1.label("WIDGET 1");
widget1.is_bordered(true);
widget1.is_shadowed(false);
widget1.is_visible (true);
widget1.label_attr(BLUE_BLACK);
widget1.border_attr(MAGENTA_BLACK);

widget2.position({CENTER, TOP});
widget2.size({51, 10});
widget2.label("WIDGET 2");
widget2.is_bordered(true);
widget2.is_shadowed(true);
widget2.is_visible (true);
widget2.label_attr(WHITE_RED);
widget2.border_attr(MAGENTA_MAGENTA);
widget2.shadow_attr(RED_RED);

// You are unable to see this widget's is_bordered,
// is_shadowed and label but it still has
// dedicated space to draw it's contents on
widget3.position({RIGHT, TOP});
widget3.size({21, 20});
widget3.label("");
widget3.is_bordered(false);
widget3.is_shadowed(false);
widget3.is_visible (true);

widget4.position({LEFT, CENTER});
widget4.size({31, 10});
widget4.label("WIDGET 4");
widget4.is_bordered(true);
widget4.is_shadowed(true);
widget4.is_visible (true);
widget4.label_attr(MAGENTA_CYAN);
widget4.border_attr(CYAN_BLUE);
widget4.shadow_attr(BLUE_WHITE);

text_box.position({CENTER, CENTER});
text_box.size({31, 20});
text_box.label("TEXT BOX");
text_box.is_bordered(true);
text_box.is_shadowed(false);
text_box.is_visible (true);
text_box.label_attr(BLACK_MAGENTA);
text_box.border_attr(MAGENTA_MAGENTA);
text_box.padding({1, 2, 1, 2});
text_box.text(
		"<!BLUE_BLACK>THIS IS SAMPLE TEXT\n"
		"    SAMPLE TEXT\n"
		"        TEXT<!NO_COLOR>\n"
		"TEXT\n"
		" TEXT\n"
		"  TEXT\n"
		"   TEXT\n"
		"    TEXT\n"
		"     TEEEEEEEEEEEEEEEE0123456789XT\n"
		"      TEXT\n"
		"         TEXT\n"
		"            TEXT\n"
		"               TEXT\n"
		"          TEXT\n"
		"         TEXT\n"
		"        TEXT"
	     );

widget6.position({RIGHT, CENTER});
widget6.size({31, 10});
widget6.label("WIDGET 6");
widget6.is_bordered(true);
widget6.is_shadowed(false);
widget6.is_visible (true);
widget6.label_attr(RED_BLACK);
widget6.border_attr(NO_COLOR);

widget7.position({LEFT, BOTTOM});
widget7.size({21, 20});
widget7.label("WIDGET 7");
widget7.is_bordered(true);
widget7.is_shadowed(true);
widget7.is_visible (true);
widget7.label_attr(NO_COLOR);
widget7.border_attr(GREEN_BLACK);
widget7.shadow_attr(YELLOW_BLACK);

widget8.position({CENTER, BOTTOM});
widget8.size({51, 10});
widget8.label("WIDGET 8");
widget8.is_bordered(false);
widget8.is_shadowed(false);
widget8.is_visible (true);
widget8.label_attr(YELLOW_BLACK);

widget9.position({RIGHT, BOTTOM});
widget9.size({21, 20});
widget9.label("WIDGET 9");
widget9.is_bordered(true);
widget9.is_shadowed(true);
widget9.is_visible (true);
widget9.label_attr(RED_BLACK);
widget9.border_attr(BLUE_BLACK);
widget9.shadow_attr(GREEN_BLACK);

*this << widget1;
*this << widget2;
*this << widget3;
*this << widget4;
*this << text_box;
*this << widget6;
*this << widget7;
*this << widget8;
*this << widget9;
}
