{
using namespace Cabs::Positions;

widget1.position({LEFT, TOP});
widget1.size({21, 20});
widget1.label("WIDGET 1");
widget1.border(true);
widget1.shadow(false);
widget1.label_attr(COLOR_PAIR(5));
widget1.border_attr(COLOR_PAIR(6));
widget1.shadow_attr(COLOR_PAIR(2));

widget2.position({CENTER, TOP});
widget2.size({51, 10});
widget2.label("WIDGET 2");
widget2.border(false);
widget2.shadow(true);
widget2.label_attr(COLOR_PAIR(16));
widget2.border_attr(COLOR_PAIR(5));
widget2.shadow_attr(COLOR_PAIR(12));

widget3.position({RIGHT, TOP});
widget3.size({21, 20});
widget3.label("WIDGET 3");
widget3.border(false);
widget3.shadow(false);
widget3.label_attr(COLOR_PAIR(8));
widget3.border_attr(COLOR_PAIR(4));
widget3.shadow_attr(COLOR_PAIR(12));

widget4.position({LEFT, CENTER});
widget4.size({31, 10});
widget4.label("WIDGET 4");
widget4.border(true);
widget4.shadow(true);
widget4.label_attr(COLOR_PAIR(54));
widget4.border_attr(COLOR_PAIR(35));
widget4.shadow_attr(COLOR_PAIR(29));

text_box.position({CENTER, CENTER});
text_box.size({31, 20});
text_box.label("TEXT BOX");
text_box.border(true);
text_box.shadow(true);
text_box.label_attr(COLOR_PAIR(41));
text_box.border_attr(COLOR_PAIR(46));
text_box.shadow_attr(COLOR_PAIR(1));
text_box.text({
		"THIS IS SAMPLE TEXT",
		"    SAMPLE TEXT",
		"        TEXT",
		"TEXT",
		" TEXT",
		"  TEXT",
		"   TEXT",
		"    TEXT",
		"     TEEEEEEEEEEEEEEEEEXT",
		"      TEXT",
		"         TEXT",
		"            TEXT",
		"               TEXT",
		"          TEXT",
		"      TEXT",
	      });

widget6.position({RIGHT, CENTER});
widget6.size({31, 10});
widget6.label("WIDGET 6");
widget6.border(true);
widget6.shadow(false);
widget6.label_attr(COLOR_PAIR(2));
widget6.border_attr(COLOR_PAIR(220));
widget6.shadow_attr(COLOR_PAIR(3));

widget7.position({LEFT, BOTTOM});
widget7.size({21, 20});
widget7.label("WIDGET 7");
widget7.border(true);
widget7.shadow(true);
widget7.label_attr(COLOR_PAIR(0));
widget7.border_attr(COLOR_PAIR(3));
widget7.shadow_attr(COLOR_PAIR(4));

widget8.position({CENTER, BOTTOM});
widget8.size({51, 10});
widget8.label("WIDGET 8");
widget8.border(false);
widget8.shadow(true);
widget8.label_attr(COLOR_PAIR(4));
widget8.border_attr(COLOR_PAIR(5));
widget8.shadow_attr(COLOR_PAIR(1));

widget9.position({RIGHT, BOTTOM});
widget9.size({21, 20});
widget9.label("WIDGET 9");
widget9.border(true);
widget9.shadow(true);
widget9.label_attr(COLOR_PAIR(2));
widget9.border_attr(COLOR_PAIR(5));
widget9.shadow_attr(COLOR_PAIR(3));

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
