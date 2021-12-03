from django.db import models



class UserSong(models.Model):
    emotion = models.CharField(max_length=100)
    song = models.CharField(max_length=250)
    url = models.CharField(max_length=400)

    def __str__(self):
        return "감정 : "+self.emotion + ", 노래 : "+ self.song
