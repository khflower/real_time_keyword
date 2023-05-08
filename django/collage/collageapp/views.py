from django.shortcuts import render

# Create your views here.
from django.views.generic import TemplateView
import pandas as pd

class AboutView(TemplateView):
    template_name = "collageapp/1.html"

    collage = pd.read_csv('collageapp/csv/collage.csv')


    def get_context_data(self, **kwargs):
        context = super(AboutView, self).get_context_data(**kwargs)

        if len(self.collage) >=1:
          context['one_title'] = self.collage['article_title'][0].replace('"',"'")
          context['one_content'] = self.collage['article_summary'][0].replace('"',"'")
          context['one_link'] = self.collage['article_link'][0]
          context['one_keyword'] = self.collage['keyword'][0]
          context['one_left'] = int(self.collage['img_x'][0])
          context['one_top'] = int(self.collage['img_y'][0])
          context['one_image'] = self.collage['img'][0]
          context['one_newsimage'] = self.collage['img_article'][0]
          context['one_rank'] = str(int(self.collage['img'][0][30])+1) +"위 : "+ self.collage['keyword'][0]
          context['time'] = self.collage['time'][0]
          


        if len(self.collage) >=2:
          context['two_title'] = self.collage['article_title'][1].replace('"',"'")
          context['two_content'] = self.collage['article_summary'][1].replace('"',"'")
          context['two_link'] = self.collage['article_link'][1]
          context['two_keyword'] = self.collage['keyword'][1]
          context['two_left'] = self.collage['img_x'][1]
          context['two_top'] = self.collage['img_y'][1]
          context['two_image'] = self.collage['img'][1]
          context['two_newsimage'] = self.collage['img_article'][1]
          context['two_rank'] = str(int(self.collage['img'][1][30])+1) +"위 : "+ self.collage['keyword'][1]


        if len(self.collage) >=3:
          context['three_title'] = self.collage['article_title'][2].replace('"',"'")
          context['three_content'] = self.collage['article_summary'][2].replace('"',"'")
          context['three_link'] = self.collage['article_link'][2]
          context['three_keyword'] = self.collage['keyword'][2]
          context['three_left'] = self.collage['img_x'][2]
          context['three_top'] = self.collage['img_y'][2]
          context['three_image'] = self.collage['img'][2]
          context['three_newsimage'] = self.collage['img_article'][2]
          context['three_rank'] = str(int(self.collage['img'][2][30])+1) +"위 : "+ self.collage['keyword'][2]

        if len(self.collage) >=4:
          context['four_title'] = self.collage['article_title'][3].replace('"',"'")
          context['four_content'] = self.collage['article_summary'][3].replace('"',"'")
          context['four_link'] = self.collage['article_link'][3]
          context['four_keyword'] = self.collage['keyword'][3]
          context['four_left'] = self.collage['img_x'][3]
          context['four_top'] = self.collage['img_y'][3]
          context['four_image'] = self.collage['img'][3]
          context['four_newsimage'] = self.collage['img_article'][3]
          context['four_rank'] = str(int(self.collage['img'][3][30])+1) +"위 : "+ self.collage['keyword'][3]


        if len(self.collage) >=5:
          context['five_title'] = self.collage['article_title'][4].replace('"',"'")
          context['five_content'] = self.collage['article_summary'][4].replace('"',"'")
          context['five_link'] = self.collage['article_link'][4]
          context['five_keyword'] = self.collage['keyword'][4]
          context['five_left'] = self.collage['img_x'][4]
          context['five_top'] = self.collage['img_y'][4]
          context['five_image'] = self.collage['img'][4]
          context['five_newsimage'] = self.collage['img_article'][4]
          context['five_rank'] = str(int(self.collage['img'][4][30])+1) +"위 : "+ self.collage['keyword'][4]
          
        

        if len(self.collage) >=6:
          context['six_title'] = self.collage['article_title'][5].replace('"',"'")
          context['six_content'] = self.collage['article_summary'][5].replace('"',"'")
          context['six_link'] = self.collage['article_link'][5]
          context['six_keyword'] = self.collage['keyword'][5]
          context['six_left'] = self.collage['img_x'][5]
          context['six_top'] = self.collage['img_y'][5]
          context['six_image'] = self.collage['img'][5]
          context['six_newsimage'] = self.collage['img_article'][5]
          context['six_rank'] = str(int(self.collage['img'][5][30])+1) +"위 : "+ self.collage['keyword'][5]


        if len(self.collage) >=7:
          context['seven_title'] = self.collage['article_title'][6].replace('"',"'")
          context['seven_content'] = self.collage['article_summary'][6].replace('"',"'")
          context['seven_link'] = self.collage['article_link'][6]
          context['seven_keyword'] = self.collage['keyword'][6]
          context['seven_left'] = self.collage['img_x'][6]
          context['seven_top'] = self.collage['img_y'][6]
          context['seven_image'] = self.collage['img'][6]
          context['seven_newsimage'] = self.collage['img_article'][6]
          context['seven_rank'] = str(int(self.collage['img'][6][30])+1) +"위 : "+ self.collage['keyword'][6]

        if len(self.collage) >=8:
          context['eight_title'] = self.collage['article_title'][7].replace('"',"'")
          context['eight_content'] = self.collage['article_summary'][7].replace('"',"'")
          context['eight_link'] = self.collage['article_link'][7]
          context['eight_keyword'] = self.collage['keyword'][7]
          context['eight_left'] = self.collage['img_x'][7]
          context['eight_top'] = self.collage['img_y'][7]
          context['eight_image'] = self.collage['img'][7]
          context['eight_newsimage'] = self.collage['img_article'][7]
          context['eight_rank'] = str(int(self.collage['img'][7][30])+1) +"위 : "+ self.collage['keyword'][7]


        if len(self.collage) >=9:
          context['nine_title'] = self.collage['article_title'][8].replace('"',"'")
          context['nine_content'] = self.collage['article_summary'][8].replace('"',"'")
          context['nine_link'] = self.collage['article_link'][8]
          context['nine_keyword'] = self.collage['keyword'][8]
          context['nine_left'] = self.collage['img_x'][8]
          context['nine_top'] = self.collage['img_y'][8]
          context['nine_image'] = self.collage['img'][8]
          context['nine_newsimage'] = self.collage['img_article'][8]
          context['nine_rank'] = str(int(self.collage['img'][8][30])+1) +"위 : "+ self.collage['keyword'][8]
          



        if len(self.collage) >=10:
          context['ten_title'] = self.collage['article_title'][9].replace('"',"'")
          context['ten_content'] = self.collage['article_summary'][9].replace('"',"'")
          context['ten_link'] = self.collage['article_link'][9]
          context['ten_keyword'] = self.collage['keyword'][9]
          context['ten_left'] = self.collage['img_x'][9]
          context['ten_top'] = self.collage['img_y'][9]
          context['ten_image'] = self.collage['img'][9]
          context['ten_newsimage'] = self.collage['img_article'][9]
          context['ten_rank'] = str(int(self.collage['img'][9][30])+1) +"위 : "+ self.collage['keyword'][9]

        
        return context






def home(request):
    keyword = pd.read_csv('collageapp/csv/keyword.csv')
    collage = pd.read_csv('collageapp/csv/collage.csv')
    return render(request, 'collageapp/2.html', {'one_keyword': keyword['keyword'][0],
                                                  'two_keyword': keyword['keyword'][1],
                                                  'three_keyword': keyword['keyword'][2],
                                                  'four_keyword': keyword['keyword'][3],
                                                  'five_keyword': keyword['keyword'][4],
                                                  'six_keyword': keyword['keyword'][5],
                                                  'seven_keyword': keyword['keyword'][6],
                                                  'eight_keyword': keyword['keyword'][7],
                                                  'nine_keyword': keyword['keyword'][8],
                                                  'ten_keyword': keyword['keyword'][9],
                                                  'time': collage['time'][0]})
                                                  
    


